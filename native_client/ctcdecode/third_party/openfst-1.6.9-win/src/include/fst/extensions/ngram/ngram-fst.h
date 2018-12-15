// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// NgramFst implements a n-gram language model based upon the LOUDS data
// structure.  Please refer to "Unary Data Structures for Language Models"
// http://research.google.com/pubs/archive/37218.pdf

#ifndef FST_EXTENSIONS_NGRAM_NGRAM_FST_H_
#define FST_EXTENSIONS_NGRAM_NGRAM_FST_H_

#include <stddef.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <fst/compat.h>
#include <fst/log.h>
#include <fstream>
#include <fst/extensions/ngram/bitmap-index.h>
#include <fst/fstlib.h>
#include <fst/mapped-file.h>

namespace fst {
template <class A>
class NGramFst;
template <class A>
class NGramFstMatcher;

// Instance data containing mutable state for bookkeeping repeated access to
// the same state.
template <class A>
struct NGramFstInst {
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;
  StateId state_;
  size_t num_futures_;
  size_t offset_;
  size_t node_;
  StateId node_state_;
  std::vector<Label> context_;
  StateId context_state_;
  NGramFstInst()
      : state_(kNoStateId),
        node_state_(kNoStateId),
        context_state_(kNoStateId) {}
};

namespace internal {

// Implementation class for LOUDS based NgramFst interface.
template <class A>
class NGramFstImpl : public FstImpl<A> {
  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;
  using FstImpl<A>::SetType;
  using FstImpl<A>::WriteHeader;

  friend class ArcIterator<NGramFst<A>>;
  friend class NGramFstMatcher<A>;

 public:
  using FstImpl<A>::InputSymbols;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::Properties;

  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;

  NGramFstImpl() {
    SetType("ngram");
    SetInputSymbols(nullptr);
    SetOutputSymbols(nullptr);
    SetProperties(kStaticProperties);
  }

  NGramFstImpl(const Fst<A> &fst, std::vector<StateId> *order_out);

  explicit NGramFstImpl(const Fst<A> &fst) : NGramFstImpl(fst, nullptr) {}

  NGramFstImpl(const NGramFstImpl &other) {
    FSTERROR() << "Copying NGramFst Impls is not supported, use safe = false.";
    SetProperties(kError, kError);
  }

  ~NGramFstImpl() override {
    if (owned_) {
      delete[] data_;
    }
  }

  static NGramFstImpl<A> *Read(std::istream &strm,  // NOLINT
                               const FstReadOptions &opts) {
    NGramFstImpl<A> *impl = new NGramFstImpl();
    FstHeader hdr;
    if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) return 0;
    uint64_t num_states, num_futures, num_final;
    const size_t offset =
        sizeof(num_states) + sizeof(num_futures) + sizeof(num_final);
    // Peek at num_states and num_futures to see how much more needs to be read.
    strm.read(reinterpret_cast<char *>(&num_states), sizeof(num_states));
    strm.read(reinterpret_cast<char *>(&num_futures), sizeof(num_futures));
    strm.read(reinterpret_cast<char *>(&num_final), sizeof(num_final));
    size_t size = Storage(num_states, num_futures, num_final);
    MappedFile *data_region = MappedFile::Allocate(size);
    char *data = reinterpret_cast<char *>(data_region->mutable_data());
    // Copy num_states, num_futures and num_final back into data.
    memcpy(data, reinterpret_cast<char *>(&num_states), sizeof(num_states));
    memcpy(data + sizeof(num_states), reinterpret_cast<char *>(&num_futures),
           sizeof(num_futures));
    memcpy(data + sizeof(num_states) + sizeof(num_futures),
           reinterpret_cast<char *>(&num_final), sizeof(num_final));
    strm.read(data + offset, size - offset);
    if (strm.fail()) {
      delete impl;
      return nullptr;
    }
    impl->Init(data, false, data_region);
    return impl;
  }

  bool Write(std::ostream &strm,  // NOLINT
             const FstWriteOptions &opts) const {
    FstHeader hdr;
    hdr.SetStart(Start());
    hdr.SetNumStates(num_states_);
    WriteHeader(strm, opts, kFileVersion, &hdr);
    strm.write(data_, StorageSize());
    return !strm.fail();
  }

  StateId Start() const { return start_; }

  Weight Final(StateId state) const {
    if (final_index_.Get(state)) {
      return final_probs_[final_index_.Rank1(state)];
    } else {
      return Weight::Zero();
    }
  }

  size_t NumArcs(StateId state, NGramFstInst<A> *inst = nullptr) const {
    if (inst == nullptr) {
      const std::pair<size_t, size_t> zeros =
          (state == 0) ? select_root_ : future_index_.Select0s(state);
      return zeros.second - zeros.first - 1;
    }
    SetInstFuture(state, inst);
    return inst->num_futures_ + ((state == 0) ? 0 : 1);
  }

  size_t NumInputEpsilons(StateId state) const {
    // State 0 has no parent, thus no backoff.
    if (state == 0) return 0;
    return 1;
  }

  size_t NumOutputEpsilons(StateId state) const {
    return NumInputEpsilons(state);
  }

  StateId NumStates() const { return num_states_; }

  void InitStateIterator(StateIteratorData<A> *data) const {
    data->base = 0;
    data->nstates = num_states_;
  }

  static size_t Storage(uint64_t num_states, uint64_t num_futures,
                        uint64_t num_final) {
    uint64_t b64;
    Weight weight;
    Label label;
    size_t offset =
        sizeof(num_states) + sizeof(num_futures) + sizeof(num_final);
    offset +=
        sizeof(b64) * (BitmapIndex::StorageSize(num_states * 2 + 1) +
                       BitmapIndex::StorageSize(num_futures + num_states + 1) +
                       BitmapIndex::StorageSize(num_states));
    offset += (num_states + 1) * sizeof(label) + num_futures * sizeof(label);
    // Pad for alignemnt, see
    // http://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
    offset = (offset + sizeof(weight) - 1) & ~(sizeof(weight) - 1);
    offset += (num_states + 1) * sizeof(weight) + num_final * sizeof(weight) +
              (num_futures + 1) * sizeof(weight);
    return offset;
  }

  void SetInstFuture(StateId state, NGramFstInst<A> *inst) const {
    if (inst->state_ != state) {
      inst->state_ = state;
      const std::pair<size_t, size_t> zeros = future_index_.Select0s(state);
      inst->num_futures_ = zeros.second - zeros.first - 1;
      inst->offset_ = future_index_.Rank1(zeros.first + 1);
    }
  }

  void SetInstNode(NGramFstInst<A> *inst) const {
    if (inst->node_state_ != inst->state_) {
      inst->node_state_ = inst->state_;
      inst->node_ = context_index_.Select1(inst->state_);
    }
  }

  void SetInstContext(NGramFstInst<A> *inst) const {
    SetInstNode(inst);
    if (inst->context_state_ != inst->state_) {
      inst->context_state_ = inst->state_;
      inst->context_.clear();
      size_t node = inst->node_;
      while (node != 0) {
        inst->context_.push_back(context_words_[context_index_.Rank1(node)]);
        node = context_index_.Select1(context_index_.Rank0(node) - 1);
      }
    }
  }

  // Access to the underlying representation
  const char *GetData(size_t *data_size) const {
    *data_size = StorageSize();
    return data_;
  }

  void Init(const char *data, bool owned, MappedFile *file = nullptr);

  const std::vector<Label> &GetContext(StateId s, NGramFstInst<A> *inst) const {
    SetInstFuture(s, inst);
    SetInstContext(inst);
    return inst->context_;
  }

  size_t StorageSize() const {
    return Storage(num_states_, num_futures_, num_final_);
  }

  void GetStates(const std::vector<Label> &context,
                 std::vector<StateId> *states) const;

 private:
  StateId Transition(const std::vector<Label> &context, Label future) const;

  // Properties always true for this Fst class.
  static const uint64_t kStaticProperties =
      kAcceptor | kIDeterministic | kODeterministic | kEpsilons | kIEpsilons |
      kOEpsilons | kILabelSorted | kOLabelSorted | kWeighted | kCyclic |
      kInitialAcyclic | kNotTopSorted | kAccessible | kCoAccessible |
      kNotString | kExpanded;
  // Current file format version.
  static const int kFileVersion = 4;
  // Minimum file format version supported.
  static const int kMinFileVersion = 4;

  std::unique_ptr<MappedFile> data_region_;
  const char *data_ = nullptr;
  bool owned_ = false;  // True if we own data_
  StateId start_ = fst::kNoStateId;
  uint64_t num_states_ = 0;
  uint64_t num_futures_ = 0;
  uint64_t num_final_ = 0;
  std::pair<size_t, size_t> select_root_;
  const Label *root_children_ = nullptr;
  // borrowed references
  const uint64_t *context_ = nullptr;
  const uint64_t *future_ = nullptr;
  const uint64_t *final_ = nullptr;
  const Label *context_words_ = nullptr;
  const Label *future_words_ = nullptr;
  const Weight *backoff_ = nullptr;
  const Weight *final_probs_ = nullptr;
  const Weight *future_probs_ = nullptr;
  BitmapIndex context_index_;
  BitmapIndex future_index_;
  BitmapIndex final_index_;
};

template <typename A>
inline void NGramFstImpl<A>::GetStates(
    const std::vector<Label> &context,
    std::vector<typename A::StateId> *states) const {
  states->clear();
  states->push_back(0);
  typename std::vector<Label>::const_reverse_iterator cit = context.rbegin();
  const Label *children = root_children_;
  size_t num_children = select_root_.second - 2;
  const Label *loc = std::lower_bound(children, children + num_children, *cit);
  if (loc == children + num_children || *loc != *cit) return;
  size_t node = 2 + loc - children;
  states->push_back(context_index_.Rank1(node));
  if (context.size() == 1) return;
  size_t node_rank = context_index_.Rank1(node);
  std::pair<size_t, size_t> zeros =
      node_rank == 0 ? select_root_ : context_index_.Select0s(node_rank);
  size_t first_child = zeros.first + 1;
  ++cit;
  if (context_index_.Get(first_child) != false) {
    size_t last_child = zeros.second - 1;
    while (cit != context.rend()) {
      children = context_words_ + context_index_.Rank1(first_child);
      loc = std::lower_bound(children, children + last_child - first_child + 1,
                             *cit);
      if (loc == children + last_child - first_child + 1 || *loc != *cit) {
        break;
      }
      ++cit;
      node = first_child + loc - children;
      states->push_back(context_index_.Rank1(node));
      node_rank = context_index_.Rank1(node);
      zeros =
          node_rank == 0 ? select_root_ : context_index_.Select0s(node_rank);
      first_child = zeros.first + 1;
      if (context_index_.Get(first_child) == false) break;
      last_child = zeros.second - 1;
    }
  }
}

}  // namespace internal

/*****************************************************************************/
template <class A>
class NGramFst : public ImplToExpandedFst<internal::NGramFstImpl<A>> {
  friend class ArcIterator<NGramFst<A>>;
  friend class NGramFstMatcher<A>;

 public:
  typedef A Arc;
  typedef typename A::StateId StateId;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef internal::NGramFstImpl<A> Impl;

  explicit NGramFst(const Fst<A> &dst)
      : ImplToExpandedFst<Impl>(std::make_shared<Impl>(dst, nullptr)) {}

  NGramFst(const Fst<A> &fst, std::vector<StateId> *order_out)
      : ImplToExpandedFst<Impl>(std::make_shared<Impl>(fst, order_out)) {}

  // Because the NGramFstImpl is a const stateless data structure, there
  // is never a need to do anything beside copy the reference.
  NGramFst(const NGramFst<A> &fst, bool safe = false)
      : ImplToExpandedFst<Impl>(fst, false) {}

  NGramFst() : ImplToExpandedFst<Impl>(std::make_shared<Impl>()) {}

  // Non-standard constructor to initialize NGramFst directly from data.
  NGramFst(const char *data, bool owned)
      : ImplToExpandedFst<Impl>(std::make_shared<Impl>()) {
    GetMutableImpl()->Init(data, owned, nullptr);
  }

  // Get method that gets the data associated with Init().
  const char *GetData(size_t *data_size) const {
    return GetImpl()->GetData(data_size);
  }

  const std::vector<Label> GetContext(StateId s) const {
    return GetImpl()->GetContext(s, &inst_);
  }

  // Consumes as much as possible of context from right to left, returns the
  // the states corresponding to the increasingly conditioned input sequence.
  void GetStates(const std::vector<Label> &context,
                 std::vector<StateId> *state) const {
    return GetImpl()->GetStates(context, state);
  }

  size_t NumArcs(StateId s) const override {
    return GetImpl()->NumArcs(s, &inst_);
  }

  NGramFst<A> *Copy(bool safe = false) const override {
    return new NGramFst(*this, safe);
  }

  static NGramFst<A> *Read(std::istream &strm, const FstReadOptions &opts) {
    Impl *impl = Impl::Read(strm, opts);
    return impl ? new NGramFst<A>(std::shared_ptr<Impl>(impl)) : nullptr;
  }

  static NGramFst<A> *Read(const string &filename) {
    if (!filename.empty()) {
      std::ifstream strm(filename,
                              std::ios_base::in | std::ios_base::binary);
      if (!strm.good()) {
        LOG(ERROR) << "NGramFst::Read: Can't open file: " << filename;
        return nullptr;
      }
      return Read(strm, FstReadOptions(filename));
    } else {
      return Read(std::cin, FstReadOptions("standard input"));
    }
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return GetImpl()->Write(strm, opts);
  }

  bool Write(const string &filename) const override {
    return Fst<A>::WriteFile(filename);
  }

  inline void InitStateIterator(StateIteratorData<A> *data) const override {
    GetImpl()->InitStateIterator(data);
  }

  inline void InitArcIterator(StateId s,
                              ArcIteratorData<A> *data) const override;

  MatcherBase<A> *InitMatcher(MatchType match_type) const override {
    return new NGramFstMatcher<A>(this, match_type);
  }

  size_t StorageSize() const { return GetImpl()->StorageSize(); }

  static bool HasRequiredProps(const Fst<A> &fst) {
    static const auto props =
        kAcceptor | kIDeterministic | kILabelSorted | kIEpsilons | kAccessible;
    return fst.Properties(props, true) == props;
  }

  static bool HasRequiredStructure(const Fst<A> &fst) {
    if (!HasRequiredProps(fst)) {
      return false;
    }
    typename A::StateId unigram = fst.Start();
    while (true) {  // Follows epsilon arc chain to find unigram state.
      if (unigram == fst::kNoStateId) return false;  // No unigram state.
      typename fst::ArcIterator<Fst<A>> aiter(fst, unigram);
      if (aiter.Done() || aiter.Value().ilabel != 0) break;
      unigram = aiter.Value().nextstate;
      aiter.Next();
    }
    // Other requirement: all states other than unigram an epsilon arc.
    for (fst::StateIterator<Fst<A>> siter(fst); !siter.Done();
         siter.Next()) {
      const typename A::StateId &state = siter.Value();
      fst::ArcIterator<Fst<A>> aiter(fst, state);
      if (state != unigram) {
        if (aiter.Done()) return false;
        if (aiter.Value().ilabel != 0) return false;
        aiter.Next();
        if (!aiter.Done() && aiter.Value().ilabel == 0) return false;
      }
    }
    return true;
  }

 private:
  using ImplToExpandedFst<Impl, ExpandedFst<A>>::GetImpl;
  using ImplToExpandedFst<Impl, ExpandedFst<A>>::GetMutableImpl;

  explicit NGramFst(std::shared_ptr<Impl> impl)
      : ImplToExpandedFst<Impl>(impl) {}

  mutable NGramFstInst<A> inst_;
};

template <class A>
inline void NGramFst<A>::InitArcIterator(StateId s,
                                         ArcIteratorData<A> *data) const {
  GetImpl()->SetInstFuture(s, &inst_);
  GetImpl()->SetInstNode(&inst_);
  data->base = new ArcIterator<NGramFst<A>>(*this, s);
}

namespace internal {

template <typename A>
NGramFstImpl<A>::NGramFstImpl(const Fst<A> &fst,
                              std::vector<StateId> *order_out) {
  typedef A Arc;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  SetType("ngram");
  SetInputSymbols(fst.InputSymbols());
  SetOutputSymbols(fst.OutputSymbols());
  SetProperties(kStaticProperties);

  // Check basic requirements for an OpenGrm language model Fst.
  if (!NGramFst<A>::HasRequiredProps(fst)) {
    FSTERROR() << "NGramFst only accepts OpenGrm language models as input";
    SetProperties(kError, kError);
    return;
  }

  int64_t num_states = CountStates(fst);
  Label *context = new Label[num_states];

  // Find the unigram state by starting from the start state, following
  // epsilons.
  StateId unigram = fst.Start();
  while (1) {
    if (unigram == kNoStateId) {
      FSTERROR() << "Could not identify unigram state";
      SetProperties(kError, kError);
      return;
    }
    ArcIterator<Fst<A>> aiter(fst, unigram);
    if (aiter.Done()) {
      LOG(WARNING) << "Unigram state " << unigram << " has no arcs.";
      break;
    }
    if (aiter.Value().ilabel != 0) break;
    unigram = aiter.Value().nextstate;
  }

  // Each state's context is determined by the subtree it is under from the
  // unigram state.
  std::queue<std::pair<StateId, Label>> label_queue;
  std::vector<bool> visited(num_states);
  // Force an epsilon link to the start state.
  label_queue.push(std::make_pair(fst.Start(), 0));
  for (ArcIterator<Fst<A>> aiter(fst, unigram); !aiter.Done(); aiter.Next()) {
    label_queue.push(
        std::make_pair(aiter.Value().nextstate, aiter.Value().ilabel));
  }
  // investigate states in breadth first fashion to assign context words.
  while (!label_queue.empty()) {
    std::pair<StateId, Label> &now = label_queue.front();
    if (!visited[now.first]) {
      context[now.first] = now.second;
      visited[now.first] = true;
      for (ArcIterator<Fst<A>> aiter(fst, now.first); !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) {
          label_queue.push(std::make_pair(arc.nextstate, now.second));
        }
      }
    }
    label_queue.pop();
  }
  visited.clear();

  // The arc from the start state should be assigned an epsilon to put it
  // in front of the all other labels (which makes Start state 1 after
  // unigram which is state 0).
  context[fst.Start()] = 0;

  // Build the tree of contexts fst by reversing the epsilon arcs from fst.
  VectorFst<Arc> context_fst;
  uint64_t num_final = 0;
  for (int i = 0; i < num_states; ++i) {
    if (fst.Final(i) != Weight::Zero()) {
      ++num_final;
    }
    context_fst.SetFinal(context_fst.AddState(), fst.Final(i));
  }
  context_fst.SetStart(unigram);
  context_fst.SetInputSymbols(fst.InputSymbols());
  context_fst.SetOutputSymbols(fst.OutputSymbols());
  int64_t num_context_arcs = 0;
  int64_t num_futures = 0;
  for (StateIterator<Fst<A>> siter(fst); !siter.Done(); siter.Next()) {
    const StateId &state = siter.Value();
    num_futures += fst.NumArcs(state) - fst.NumInputEpsilons(state);
    ArcIterator<Fst<A>> aiter(fst, state);
    if (!aiter.Done()) {
      const Arc &arc = aiter.Value();
      // this arc goes from state to arc.nextstate, so create an arc from
      // arc.nextstate to state to reverse it.
      if (arc.ilabel == 0) {
        context_fst.AddArc(arc.nextstate, Arc(context[state], context[state],
                                              arc.weight, state));
        num_context_arcs++;
      }
    }
  }
  if (num_context_arcs != context_fst.NumStates() - 1) {
    FSTERROR() << "Number of contexts arcs != number of states - 1";
    SetProperties(kError, kError);
    return;
  }
  if (context_fst.NumStates() != num_states) {
    FSTERROR() << "Number of contexts != number of states";
    SetProperties(kError, kError);
    return;
  }
  int64_t context_props =
      context_fst.Properties(kIDeterministic | kILabelSorted, true);
  if (!(context_props & kIDeterministic)) {
    FSTERROR() << "Input Fst is not structured properly";
    SetProperties(kError, kError);
    return;
  }
  if (!(context_props & kILabelSorted)) {
    ArcSort(&context_fst, ILabelCompare<Arc>());
  }

  delete[] context;

  uint64_t b64;
  Weight weight;
  Label label = kNoLabel;
  const size_t storage = Storage(num_states, num_futures, num_final);
  MappedFile *data_region = MappedFile::Allocate(storage);
  char *data = reinterpret_cast<char *>(data_region->mutable_data());
  memset(data, 0, storage);
  size_t offset = 0;
  memcpy(data + offset, reinterpret_cast<char *>(&num_states),
         sizeof(num_states));
  offset += sizeof(num_states);
  memcpy(data + offset, reinterpret_cast<char *>(&num_futures),
         sizeof(num_futures));
  offset += sizeof(num_futures);
  memcpy(data + offset, reinterpret_cast<char *>(&num_final),
         sizeof(num_final));
  offset += sizeof(num_final);
  uint64_t *context_bits = reinterpret_cast<uint64_t *>(data + offset);
  offset += BitmapIndex::StorageSize(num_states * 2 + 1) * sizeof(b64);
  uint64_t *future_bits = reinterpret_cast<uint64_t *>(data + offset);
  offset +=
      BitmapIndex::StorageSize(num_futures + num_states + 1) * sizeof(b64);
  uint64_t *final_bits = reinterpret_cast<uint64_t *>(data + offset);
  offset += BitmapIndex::StorageSize(num_states) * sizeof(b64);
  Label *context_words = reinterpret_cast<Label *>(data + offset);
  offset += (num_states + 1) * sizeof(label);
  Label *future_words = reinterpret_cast<Label *>(data + offset);
  offset += num_futures * sizeof(label);
  offset = (offset + sizeof(weight) - 1) & ~(sizeof(weight) - 1);
  Weight *backoff = reinterpret_cast<Weight *>(data + offset);
  offset += (num_states + 1) * sizeof(weight);
  Weight *final_probs = reinterpret_cast<Weight *>(data + offset);
  offset += num_final * sizeof(weight);
  Weight *future_probs = reinterpret_cast<Weight *>(data + offset);
  int64_t context_arc = 0, future_arc = 0, context_bit = 0, future_bit = 0,
        final_bit = 0;

  // pseudo-root bits
  BitmapIndex::Set(context_bits, context_bit++);
  ++context_bit;
  context_words[context_arc] = label;
  backoff[context_arc] = Weight::Zero();
  context_arc++;

  ++future_bit;
  if (order_out) {
    order_out->clear();
    order_out->resize(num_states);
  }

  std::queue<StateId> context_q;
  context_q.push(context_fst.Start());
  StateId state_number = 0;
  while (!context_q.empty()) {
    const StateId &state = context_q.front();
    if (order_out) {
      (*order_out)[state] = state_number;
    }

    const Weight final_weight = context_fst.Final(state);
    if (final_weight != Weight::Zero()) {
      BitmapIndex::Set(final_bits, state_number);
      final_probs[final_bit] = final_weight;
      ++final_bit;
    }

    for (ArcIterator<VectorFst<A>> aiter(context_fst, state); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      context_words[context_arc] = arc.ilabel;
      backoff[context_arc] = arc.weight;
      ++context_arc;
      BitmapIndex::Set(context_bits, context_bit++);
      context_q.push(arc.nextstate);
    }
    ++context_bit;

    for (ArcIterator<Fst<A>> aiter(fst, state); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {
        future_words[future_arc] = arc.ilabel;
        future_probs[future_arc] = arc.weight;
        ++future_arc;
        BitmapIndex::Set(future_bits, future_bit++);
      }
    }
    ++future_bit;
    ++state_number;
    context_q.pop();
  }

  if ((state_number != num_states) || (context_bit != num_states * 2 + 1) ||
      (context_arc != num_states) || (future_arc != num_futures) ||
      (future_bit != num_futures + num_states + 1) ||
      (final_bit != num_final)) {
    FSTERROR() << "Structure problems detected during construction";
    SetProperties(kError, kError);
    return;
  }

  Init(data, false, data_region);
}

template <typename A>
inline void NGramFstImpl<A>::Init(const char *data, bool owned,
                                  MappedFile *data_region) {
  if (owned_) {
    delete[] data_;
  }
  data_region_.reset(data_region);
  owned_ = owned;
  data_ = data;
  size_t offset = 0;
  num_states_ = *(reinterpret_cast<const uint64_t *>(data_ + offset));
  offset += sizeof(num_states_);
  num_futures_ = *(reinterpret_cast<const uint64_t *>(data_ + offset));
  offset += sizeof(num_futures_);
  num_final_ = *(reinterpret_cast<const uint64_t *>(data_ + offset));
  offset += sizeof(num_final_);
  uint64_t bits;
  size_t context_bits = num_states_ * 2 + 1;
  size_t future_bits = num_futures_ + num_states_ + 1;
  context_ = reinterpret_cast<const uint64_t *>(data_ + offset);
  offset += BitmapIndex::StorageSize(context_bits) * sizeof(bits);
  future_ = reinterpret_cast<const uint64_t *>(data_ + offset);
  offset += BitmapIndex::StorageSize(future_bits) * sizeof(bits);
  final_ = reinterpret_cast<const uint64_t *>(data_ + offset);
  offset += BitmapIndex::StorageSize(num_states_) * sizeof(bits);
  context_words_ = reinterpret_cast<const Label *>(data_ + offset);
  offset += (num_states_ + 1) * sizeof(*context_words_);
  future_words_ = reinterpret_cast<const Label *>(data_ + offset);
  offset += num_futures_ * sizeof(*future_words_);
  offset = (offset + sizeof(*backoff_) - 1) & ~(sizeof(*backoff_) - 1);
  backoff_ = reinterpret_cast<const Weight *>(data_ + offset);
  offset += (num_states_ + 1) * sizeof(*backoff_);
  final_probs_ = reinterpret_cast<const Weight *>(data_ + offset);
  offset += num_final_ * sizeof(*final_probs_);
  future_probs_ = reinterpret_cast<const Weight *>(data_ + offset);

  context_index_.BuildIndex(context_, context_bits);
  future_index_.BuildIndex(future_, future_bits);
  final_index_.BuildIndex(final_, num_states_);

  select_root_ = context_index_.Select0s(0);
  if (context_index_.Rank1(0) != 0 || select_root_.first != 1 ||
      context_index_.Get(2) == false) {
    FSTERROR() << "Malformed file";
    SetProperties(kError, kError);
    return;
  }
  root_children_ = context_words_ + context_index_.Rank1(2);
  start_ = 1;
}

template <typename A>
inline typename A::StateId NGramFstImpl<A>::Transition(
    const std::vector<Label> &context, Label future) const {
  const Label *children = root_children_;
  size_t num_children = select_root_.second - 2;
  const Label *loc =
      std::lower_bound(children, children + num_children, future);
  if (loc == children + num_children || *loc != future) {
    return context_index_.Rank1(0);
  }
  size_t node = 2 + loc - children;
  size_t node_rank = context_index_.Rank1(node);
  std::pair<size_t, size_t> zeros =
      (node_rank == 0) ? select_root_ : context_index_.Select0s(node_rank);
  size_t first_child = zeros.first + 1;
  if (context_index_.Get(first_child) == false) {
    return context_index_.Rank1(node);
  }
  size_t last_child = zeros.second - 1;
  for (int word = context.size() - 1; word >= 0; --word) {
    children = context_words_ + context_index_.Rank1(first_child);
    loc = std::lower_bound(children, children + last_child - first_child + 1,
                           context[word]);
    if (loc == children + last_child - first_child + 1 ||
        *loc != context[word]) {
      break;
    }
    node = first_child + loc - children;
    node_rank = context_index_.Rank1(node);
    zeros =
        (node_rank == 0) ? select_root_ : context_index_.Select0s(node_rank);
    first_child = zeros.first + 1;
    if (context_index_.Get(first_child) == false) break;
    last_child = zeros.second - 1;
  }
  return context_index_.Rank1(node);
}

}  // namespace internal

/*****************************************************************************/
template <class A>
class NGramFstMatcher : public MatcherBase<A> {
 public:
  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;

  // This makes a copy of the FST.
  NGramFstMatcher(const NGramFst<A> &fst, MatchType match_type)
      : owned_fst_(fst.Copy()),
        fst_(*owned_fst_),
        inst_(fst_.inst_),
        match_type_(match_type),
        current_loop_(false),
        loop_(kNoLabel, 0, A::Weight::One(), kNoStateId) {
    if (match_type_ == MATCH_OUTPUT) {
      std::swap(loop_.ilabel, loop_.olabel);
    }
  }

  // This doesn't copy the FST.
  NGramFstMatcher(const NGramFst<A> *fst, MatchType match_type)
      : fst_(*fst),
        inst_(fst_.inst_),
        match_type_(match_type),
        current_loop_(false),
        loop_(kNoLabel, 0, A::Weight::One(), kNoStateId) {
    if (match_type_ == MATCH_OUTPUT) {
      std::swap(loop_.ilabel, loop_.olabel);
    }
  }

  // This makes a copy of the FST.
  NGramFstMatcher(const NGramFstMatcher<A> &matcher, bool safe = false)
      : owned_fst_(matcher.fst_.Copy(safe)),
        fst_(*owned_fst_),
        inst_(matcher.inst_),
        match_type_(matcher.match_type_),
        current_loop_(false),
        loop_(kNoLabel, 0, A::Weight::One(), kNoStateId) {
    if (match_type_ == MATCH_OUTPUT) {
      std::swap(loop_.ilabel, loop_.olabel);
    }
  }

  NGramFstMatcher<A> *Copy(bool safe = false) const override {
    return new NGramFstMatcher<A>(*this, safe);
  }

  MatchType Type(bool test) const override { return match_type_; }

  const Fst<A> &GetFst() const override { return fst_; }

  uint64_t Properties(uint64_t props) const override { return props; }

  void SetState(StateId s) final {
    fst_.GetImpl()->SetInstFuture(s, &inst_);
    current_loop_ = false;
  }

  bool Find(Label label) final {
    const Label nolabel = kNoLabel;
    done_ = true;
    if (label == 0 || label == nolabel) {
      if (label == 0) {
        current_loop_ = true;
        loop_.nextstate = inst_.state_;
      }
      // The unigram state has no epsilon arc.
      if (inst_.state_ != 0) {
        arc_.ilabel = arc_.olabel = 0;
        fst_.GetImpl()->SetInstNode(&inst_);
        arc_.nextstate = fst_.GetImpl()->context_index_.Rank1(
            fst_.GetImpl()->context_index_.Select1(
                fst_.GetImpl()->context_index_.Rank0(inst_.node_) - 1));
        arc_.weight = fst_.GetImpl()->backoff_[inst_.state_];
        done_ = false;
      }
    } else {
      current_loop_ = false;
      const Label *start = fst_.GetImpl()->future_words_ + inst_.offset_;
      const Label *end = start + inst_.num_futures_;
      const Label *search = std::lower_bound(start, end, label);
      if (search != end && *search == label) {
        size_t state = search - start;
        arc_.ilabel = arc_.olabel = label;
        arc_.weight = fst_.GetImpl()->future_probs_[inst_.offset_ + state];
        fst_.GetImpl()->SetInstContext(&inst_);
        arc_.nextstate = fst_.GetImpl()->Transition(inst_.context_, label);
        done_ = false;
      }
    }
    return !Done();
  }

  bool Done() const final { return !current_loop_ && done_; }

  const Arc &Value() const final { return (current_loop_) ? loop_ : arc_; }

  void Next() final {
    if (current_loop_) {
      current_loop_ = false;
    } else {
      done_ = true;
    }
  }

  std::ptrdiff_t Priority(StateId s) final { return fst_.NumArcs(s); }

 private:
  std::unique_ptr<NGramFst<A>> owned_fst_;
  const NGramFst<A> &fst_;
  NGramFstInst<A> inst_;
  MatchType match_type_;  // Supplied by caller
  bool done_;
  Arc arc_;
  bool current_loop_;  // Current arc is the implicit loop
  Arc loop_;
};

/*****************************************************************************/
// Specialization for NGramFst; see generic version in fst.h
// for sample usage (but use the ProdLmFst type!). This version
// should inline.
template <class A>
class StateIterator<NGramFst<A>> : public StateIteratorBase<A> {
 public:
  typedef typename A::StateId StateId;

  explicit StateIterator(const NGramFst<A> &fst)
      : s_(0), num_states_(fst.NumStates()) {}

  bool Done() const final { return s_ >= num_states_; }

  StateId Value() const final { return s_; }

  void Next() final { ++s_; }

  void Reset() final { s_ = 0; }

 private:
  StateId s_;
  StateId num_states_;
};

/*****************************************************************************/
template <class A>
class ArcIterator<NGramFst<A>> : public ArcIteratorBase<A> {
 public:
  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;

  ArcIterator(const NGramFst<A> &fst, StateId state)
      : lazy_(~0), impl_(fst.GetImpl()), i_(0), flags_(kArcValueFlags) {
    inst_ = fst.inst_;
    impl_->SetInstFuture(state, &inst_);
    impl_->SetInstNode(&inst_);
  }

  bool Done() const final {
    return i_ >=
           ((inst_.node_ == 0) ? inst_.num_futures_ : inst_.num_futures_ + 1);
  }

  const Arc &Value() const final {
    bool eps = (inst_.node_ != 0 && i_ == 0);
    StateId state = (inst_.node_ == 0) ? i_ : i_ - 1;
    if (flags_ & lazy_ & (kArcILabelValue | kArcOLabelValue)) {
      arc_.ilabel = arc_.olabel =
          eps ? 0 : impl_->future_words_[inst_.offset_ + state];
      lazy_ &= ~(kArcILabelValue | kArcOLabelValue);
    }
    if (flags_ & lazy_ & kArcNextStateValue) {
      if (eps) {
        arc_.nextstate =
            impl_->context_index_.Rank1(impl_->context_index_.Select1(
                impl_->context_index_.Rank0(inst_.node_) - 1));
      } else {
        if (lazy_ & kArcNextStateValue) {
          impl_->SetInstContext(&inst_);  // first time only.
        }
        arc_.nextstate = impl_->Transition(
            inst_.context_, impl_->future_words_[inst_.offset_ + state]);
      }
      lazy_ &= ~kArcNextStateValue;
    }
    if (flags_ & lazy_ & kArcWeightValue) {
      arc_.weight = eps ? impl_->backoff_[inst_.state_]
                        : impl_->future_probs_[inst_.offset_ + state];
      lazy_ &= ~kArcWeightValue;
    }
    return arc_;
  }

  void Next() final {
    ++i_;
    lazy_ = ~0;
  }

  size_t Position() const final { return i_; }

  void Reset() final {
    i_ = 0;
    lazy_ = ~0;
  }

  void Seek(size_t a) final {
    if (i_ != a) {
      i_ = a;
      lazy_ = ~0;
    }
  }

  uint32_t Flags() const final { return flags_; }

  void SetFlags(uint32_t flags, uint32_t mask) final {
    flags_ &= ~mask;
    flags_ |= (flags & kArcValueFlags);
  }

 private:
  mutable Arc arc_;
  mutable uint32_t lazy_;
  const internal::NGramFstImpl<A> *impl_;  // Borrowed reference.
  mutable NGramFstInst<A> inst_;

  size_t i_;
  uint32_t flags_;
};

}  // namespace fst
#endif  // FST_EXTENSIONS_NGRAM_NGRAM_FST_H_
