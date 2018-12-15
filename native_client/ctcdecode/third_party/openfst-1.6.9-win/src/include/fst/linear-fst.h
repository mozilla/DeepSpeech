// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes for building, storing and representing log-linear models as FSTs.

#ifndef FST_EXTENSIONS_LINEAR_LINEAR_FST_H_
#define FST_EXTENSIONS_LINEAR_LINEAR_FST_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <fst/compat.h>
#include <fst/log.h>
#include <fst/extensions/pdt/collection.h>
#include <fst/bi-table.h>
#include <fst/cache.h>
#include <fstream>
#include <fst/fst.h>
#include <fst/matcher.h>
#include <fst/symbol-table.h>

#include <fst/extensions/linear/linear-fst-data.h>

namespace fst {

// Forward declaration of the specialized matcher for both
// LinearTaggerFst and LinearClassifierFst.
template <class F>
class LinearFstMatcherTpl;

namespace internal {

// Implementation class for on-the-fly generated LinearTaggerFst with
// special optimization in matching.
template <class A>
class LinearTaggerFstImpl : public CacheImpl<A> {
 public:
  using FstImpl<A>::SetType;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;
  using FstImpl<A>::WriteHeader;

  using CacheBaseImpl<CacheState<A>>::PushArc;
  using CacheBaseImpl<CacheState<A>>::HasArcs;
  using CacheBaseImpl<CacheState<A>>::HasFinal;
  using CacheBaseImpl<CacheState<A>>::HasStart;
  using CacheBaseImpl<CacheState<A>>::SetArcs;
  using CacheBaseImpl<CacheState<A>>::SetFinal;
  using CacheBaseImpl<CacheState<A>>::SetStart;

  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef typename Collection<StateId, Label>::SetIterator NGramIterator;

  // Constructs an empty FST by default.
  LinearTaggerFstImpl()
      : CacheImpl<A>(CacheOptions()),
        data_(std::make_shared<LinearFstData<A>>()),
        delay_(0) {
    SetType("linear-tagger");
  }

  // Constructs the FST with given data storage and symbol
  // tables.
  //
  // TODO(wuke): when there is no constraint on output we can delay
  // less than `data->MaxFutureSize` positions.
  LinearTaggerFstImpl(const LinearFstData<Arc> *data, const SymbolTable *isyms,
                      const SymbolTable *osyms, CacheOptions opts)
      : CacheImpl<A>(opts), data_(data), delay_(data->MaxFutureSize()) {
    SetType("linear-tagger");
    SetProperties(kILabelSorted, kFstProperties);
    SetInputSymbols(isyms);
    SetOutputSymbols(osyms);
    ReserveStubSpace();
  }

  // Copy by sharing the underlying data storage.
  LinearTaggerFstImpl(const LinearTaggerFstImpl &impl)
      : CacheImpl<A>(impl), data_(impl.data_), delay_(impl.delay_) {
    SetType("linear-tagger");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
    ReserveStubSpace();
  }

  StateId Start() {
    if (!HasStart()) {
      StateId start = FindStartState();
      SetStart(start);
    }
    return CacheImpl<A>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      state_stub_.clear();
      FillState(s, &state_stub_);
      if (CanBeFinal(state_stub_))
        SetFinal(s, data_->FinalWeight(InternalBegin(state_stub_),
                                       InternalEnd(state_stub_)));
      else
        SetFinal(s, Weight::Zero());
    }
    return CacheImpl<A>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<A>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<A>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<A>::NumOutputEpsilons(s);
  }

  void InitArcIterator(StateId s, ArcIteratorData<A> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl<A>::InitArcIterator(s, data);
  }

  // Computes the outgoing transitions from a state, creating new
  // destination states as needed.
  void Expand(StateId s);

  // Appends to `arcs` all out-going arcs from state `s` that matches `label` as
  // the input label.
  void MatchInput(StateId s, Label ilabel, std::vector<Arc> *arcs);

  static LinearTaggerFstImpl *Read(std::istream &strm,
                                   const FstReadOptions &opts);

  bool Write(std::ostream &strm,  // NOLINT
             const FstWriteOptions &opts) const {
    FstHeader header;
    header.SetStart(kNoStateId);
    WriteHeader(strm, opts, kFileVersion, &header);
    data_->Write(strm);
    if (!strm) {
      LOG(ERROR) << "LinearTaggerFst::Write: Write failed: " << opts.source;
      return false;
    }
    return true;
  }

 private:
  static const int kMinFileVersion;
  static const int kFileVersion;

  // A collection of functions to access parts of the state tuple. A
  // state tuple is a vector of `Label`s with two parts:
  //   [buffer] [internal].
  //
  // - [buffer] is a buffer of observed input labels with length
  // `delay_`. `LinearFstData<A>::kStartOfSentence`
  // (resp. `LinearFstData<A>::kEndOfSentence`) are used as
  // paddings when the buffer has fewer than `delay_` elements, which
  // can only appear as the prefix (resp. suffix) of the buffer.
  //
  // - [internal] is the internal state tuple for `LinearFstData`
  typename std::vector<Label>::const_iterator BufferBegin(
      const std::vector<Label> &state) const {
    return state.begin();
  }

  typename std::vector<Label>::const_iterator BufferEnd(
      const std::vector<Label> &state) const {
    return state.begin() + delay_;
  }

  typename std::vector<Label>::const_iterator InternalBegin(
      const std::vector<Label> &state) const {
    return state.begin() + delay_;
  }

  typename std::vector<Label>::const_iterator InternalEnd(
      const std::vector<Label> &state) const {
    return state.end();
  }

  // The size of state tuples are fixed, reserve them in stubs
  void ReserveStubSpace() {
    state_stub_.reserve(delay_ + data_->NumGroups());
    next_stub_.reserve(delay_ + data_->NumGroups());
  }

  // Computes the start state tuple and maps it to the start state id.
  StateId FindStartState() {
    // Empty buffer with start-of-sentence paddings
    state_stub_.clear();
    state_stub_.resize(delay_, LinearFstData<A>::kStartOfSentence);
    // Append internal states
    data_->EncodeStartState(&state_stub_);
    return FindState(state_stub_);
  }

  // Tests whether the buffer in `(begin, end)` is empty.
  bool IsEmptyBuffer(typename std::vector<Label>::const_iterator begin,
                     typename std::vector<Label>::const_iterator end) const {
    // The following is guanranteed by `ShiftBuffer()`:
    // - buffer[i] == LinearFstData<A>::kEndOfSentence =>
    //       buffer[i+x] == LinearFstData<A>::kEndOfSentence
    // - buffer[i] == LinearFstData<A>::kStartOfSentence =>
    //       buffer[i-x] == LinearFstData<A>::kStartOfSentence
    return delay_ == 0 || *(end - 1) == LinearFstData<A>::kStartOfSentence ||
           *begin == LinearFstData<A>::kEndOfSentence;
  }

  // Tests whether the given state tuple can be a final state. A state
  // is final iff there is no observed input in the buffer.
  bool CanBeFinal(const std::vector<Label> &state) {
    return IsEmptyBuffer(BufferBegin(state), BufferEnd(state));
  }

  // Finds state corresponding to an n-gram. Creates new state if n-gram not
  // found.
  StateId FindState(const std::vector<Label> &ngram) {
    StateId sparse = ngrams_.FindId(ngram, true);
    StateId dense = condensed_.FindId(sparse, true);
    return dense;
  }

  // Appends after `output` the state tuple corresponding to the state id. The
  // state id must exist.
  void FillState(StateId s, std::vector<Label> *output) {
    s = condensed_.FindEntry(s);
    for (NGramIterator it = ngrams_.FindSet(s); !it.Done(); it.Next()) {
      Label label = it.Element();
      output->push_back(label);
    }
  }

  // Shifts the buffer in `state` by appending `ilabel` and popping
  // the one in the front as the return value. `next_stub_` is a
  // shifted buffer of size `delay_` where the first `delay_ - 1`
  // elements are the last `delay_ - 1` elements in the buffer of
  // `state`. The last (if any) element in `next_stub_` will be
  // `ilabel` after the call returns.
  Label ShiftBuffer(const std::vector<Label> &state, Label ilabel,
                    std::vector<Label> *next_stub_);

  // Builds an arc from state tuple `state` consuming `ilabel` and
  // `olabel`. `next_stub_` is the buffer filled in `ShiftBuffer`.
  Arc MakeArc(const std::vector<Label> &state, Label ilabel, Label olabel,
              std::vector<Label> *next_stub_);

  // Expands arcs from state `s`, equivalent to state tuple `state`,
  // with input `ilabel`. `next_stub_` is the buffer filled in
  // `ShiftBuffer`.
  void ExpandArcs(StateId s, const std::vector<Label> &state, Label ilabel,
                  std::vector<Label> *next_stub_);

  // Appends arcs from state `s`, equivalent to state tuple `state`,
  // with input `ilabel` to `arcs`. `next_stub_` is the buffer filled
  // in `ShiftBuffer`.
  void AppendArcs(StateId s, const std::vector<Label> &state, Label ilabel,
                  std::vector<Label> *next_stub_, std::vector<Arc> *arcs);

  std::shared_ptr<const LinearFstData<A>> data_;
  size_t delay_;
  // Mapping from internal state tuple to *non-consecutive* ids
  Collection<StateId, Label> ngrams_;
  // Mapping from non-consecutive id to actual state id
  CompactHashBiTable<StateId, StateId, std::hash<StateId>> condensed_;
  // Two frequently used vectors, reuse to avoid repeated heap
  // allocation
  std::vector<Label> state_stub_, next_stub_;

  LinearTaggerFstImpl &operator=(const LinearTaggerFstImpl &) = delete;
};

template <class A>
const int LinearTaggerFstImpl<A>::kMinFileVersion = 1;

template <class A>
const int LinearTaggerFstImpl<A>::kFileVersion = 1;

template <class A>
inline typename A::Label LinearTaggerFstImpl<A>::ShiftBuffer(
    const std::vector<Label> &state, Label ilabel,
    std::vector<Label> *next_stub_) {
  DCHECK(ilabel > 0 || ilabel == LinearFstData<A>::kEndOfSentence);
  if (delay_ == 0) {
    DCHECK_GT(ilabel, 0);
    return ilabel;
  } else {
    (*next_stub_)[BufferEnd(*next_stub_) - next_stub_->begin() - 1] = ilabel;
    return *BufferBegin(state);
  }
}

template <class A>
inline A LinearTaggerFstImpl<A>::MakeArc(const std::vector<Label> &state,
                                         Label ilabel, Label olabel,
                                         std::vector<Label> *next_stub_) {
  DCHECK(ilabel > 0 || ilabel == LinearFstData<A>::kEndOfSentence);
  DCHECK(olabel > 0 || olabel == LinearFstData<A>::kStartOfSentence);
  Weight weight(Weight::One());
  data_->TakeTransition(BufferEnd(state), InternalBegin(state),
                        InternalEnd(state), ilabel, olabel, next_stub_,
                        &weight);
  StateId nextstate = FindState(*next_stub_);
  // Restore `next_stub_` to its size before the call
  next_stub_->resize(delay_);
  // In the actual arc, we use epsilons instead of boundaries.
  return A(ilabel == LinearFstData<A>::kEndOfSentence ? 0 : ilabel,
           olabel == LinearFstData<A>::kStartOfSentence ? 0 : olabel, weight,
           nextstate);
}

template <class A>
inline void LinearTaggerFstImpl<A>::ExpandArcs(StateId s,
                                               const std::vector<Label> &state,
                                               Label ilabel,
                                               std::vector<Label> *next_stub_) {
  // Input label to constrain the output with, observed `delay_` steps
  // back. `ilabel` is the input label to be put on the arc, which
  // fires features.
  Label obs_ilabel = ShiftBuffer(state, ilabel, next_stub_);
  if (obs_ilabel == LinearFstData<A>::kStartOfSentence) {
    // This happens when input is shorter than `delay_`.
    PushArc(s, MakeArc(state, ilabel, LinearFstData<A>::kStartOfSentence,
                       next_stub_));
  } else {
    std::pair<typename std::vector<typename A::Label>::const_iterator,
              typename std::vector<typename A::Label>::const_iterator> range =
        data_->PossibleOutputLabels(obs_ilabel);
    for (typename std::vector<typename A::Label>::const_iterator it =
             range.first;
         it != range.second; ++it)
      PushArc(s, MakeArc(state, ilabel, *it, next_stub_));
  }
}

// TODO(wuke): this has much in duplicate with `ExpandArcs()`
template <class A>
inline void LinearTaggerFstImpl<A>::AppendArcs(StateId /*s*/,
                                               const std::vector<Label> &state,
                                               Label ilabel,
                                               std::vector<Label> *next_stub_,
                                               std::vector<Arc> *arcs) {
  // Input label to constrain the output with, observed `delay_` steps
  // back. `ilabel` is the input label to be put on the arc, which
  // fires features.
  Label obs_ilabel = ShiftBuffer(state, ilabel, next_stub_);
  if (obs_ilabel == LinearFstData<A>::kStartOfSentence) {
    // This happens when input is shorter than `delay_`.
    arcs->push_back(
        MakeArc(state, ilabel, LinearFstData<A>::kStartOfSentence, next_stub_));
  } else {
    std::pair<typename std::vector<typename A::Label>::const_iterator,
              typename std::vector<typename A::Label>::const_iterator> range =
        data_->PossibleOutputLabels(obs_ilabel);
    for (typename std::vector<typename A::Label>::const_iterator it =
             range.first;
         it != range.second; ++it)
      arcs->push_back(MakeArc(state, ilabel, *it, next_stub_));
  }
}

template <class A>
void LinearTaggerFstImpl<A>::Expand(StateId s) {
  VLOG(3) << "Expand " << s;
  state_stub_.clear();
  FillState(s, &state_stub_);

  // Precompute the first `delay_ - 1` elements in the buffer of
  // next states, which are identical for different input/output.
  next_stub_.clear();
  next_stub_.resize(delay_);
  if (delay_ > 0)
    std::copy(BufferBegin(state_stub_) + 1, BufferEnd(state_stub_),
              next_stub_.begin());

  // Epsilon transition for flushing out the next observed input
  if (!IsEmptyBuffer(BufferBegin(state_stub_), BufferEnd(state_stub_)))
    ExpandArcs(s, state_stub_, LinearFstData<A>::kEndOfSentence, &next_stub_);

  // Non-epsilon input when we haven't flushed
  if (delay_ == 0 ||
      *(BufferEnd(state_stub_) - 1) != LinearFstData<A>::kEndOfSentence)
    for (Label ilabel = data_->MinInputLabel();
         ilabel <= data_->MaxInputLabel(); ++ilabel)
      ExpandArcs(s, state_stub_, ilabel, &next_stub_);

  SetArcs(s);
}

template <class A>
void LinearTaggerFstImpl<A>::MatchInput(StateId s, Label ilabel,
                                        std::vector<Arc> *arcs) {
  state_stub_.clear();
  FillState(s, &state_stub_);

  // Precompute the first `delay_ - 1` elements in the buffer of
  // next states, which are identical for different input/output.
  next_stub_.clear();
  next_stub_.resize(delay_);
  if (delay_ > 0)
    std::copy(BufferBegin(state_stub_) + 1, BufferEnd(state_stub_),
              next_stub_.begin());

  if (ilabel == 0) {
    // Epsilon transition for flushing out the next observed input
    if (!IsEmptyBuffer(BufferBegin(state_stub_), BufferEnd(state_stub_)))
      AppendArcs(s, state_stub_, LinearFstData<A>::kEndOfSentence, &next_stub_,
                 arcs);
  } else {
    // Non-epsilon input when we haven't flushed
    if (delay_ == 0 ||
        *(BufferEnd(state_stub_) - 1) != LinearFstData<A>::kEndOfSentence)
      AppendArcs(s, state_stub_, ilabel, &next_stub_, arcs);
  }
}

template <class A>
inline LinearTaggerFstImpl<A> *LinearTaggerFstImpl<A>::Read(
    std::istream &strm, const FstReadOptions &opts) {  // NOLINT
  std::unique_ptr<LinearTaggerFstImpl<A>> impl(new LinearTaggerFstImpl<A>());
  FstHeader header;
  if (!impl->ReadHeader(strm, opts, kMinFileVersion, &header)) {
    return nullptr;
  }
  impl->data_ = std::shared_ptr<LinearFstData<A>>(LinearFstData<A>::Read(strm));
  if (!impl->data_) {
    return nullptr;
  }
  impl->delay_ = impl->data_->MaxFutureSize();
  impl->ReserveStubSpace();
  return impl.release();
}

}  // namespace internal

// This class attaches interface to implementation and handles
// reference counting, delegating most methods to ImplToFst.
template <class A>
class LinearTaggerFst : public ImplToFst<internal::LinearTaggerFstImpl<A>> {
 public:
  friend class ArcIterator<LinearTaggerFst<A>>;
  friend class StateIterator<LinearTaggerFst<A>>;
  friend class LinearFstMatcherTpl<LinearTaggerFst<A>>;

  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef DefaultCacheStore<A> Store;
  typedef typename Store::State State;
  using Impl = internal::LinearTaggerFstImpl<A>;

  LinearTaggerFst() : ImplToFst<Impl>(std::make_shared<Impl>()) {}

  explicit LinearTaggerFst(LinearFstData<A> *data,
                           const SymbolTable *isyms = nullptr,
                           const SymbolTable *osyms = nullptr,
                           CacheOptions opts = CacheOptions())
      : ImplToFst<Impl>(std::make_shared<Impl>(data, isyms, osyms, opts)) {}

  explicit LinearTaggerFst(const Fst<A> &fst)
      : ImplToFst<Impl>(std::make_shared<Impl>()) {
    LOG(FATAL) << "LinearTaggerFst: no constructor from arbitrary FST.";
  }

  // See Fst<>::Copy() for doc.
  LinearTaggerFst(const LinearTaggerFst<A> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this LinearTaggerFst. See Fst<>::Copy() for further doc.
  LinearTaggerFst<A> *Copy(bool safe = false) const override {
    return new LinearTaggerFst<A>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<A> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<A> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  MatcherBase<A> *InitMatcher(MatchType match_type) const override {
    return new LinearFstMatcherTpl<LinearTaggerFst<A>>(this, match_type);
  }

  static LinearTaggerFst<A> *Read(const string &filename) {
    if (!filename.empty()) {
      std::ifstream strm(filename,
                              std::ios_base::in | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "LinearTaggerFst::Read: Can't open file: " << filename;
        return nullptr;
      }
      return Read(strm, FstReadOptions(filename));
    } else {
      return Read(std::cin, FstReadOptions("standard input"));
    }
  }

  static LinearTaggerFst<A> *Read(std::istream &in,  // NOLINT
                                  const FstReadOptions &opts) {
    auto *impl = Impl::Read(in, opts);
    return impl ? new LinearTaggerFst<A>(std::shared_ptr<Impl>(impl)) : nullptr;
  }

  bool Write(const string &filename) const override {
    if (!filename.empty()) {
      std::ofstream strm(filename,
                               std::ios_base::out | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "LinearTaggerFst::Write: Can't open file: " << filename;
        return false;
      }
      return Write(strm, FstWriteOptions(filename));
    } else {
      return Write(std::cout, FstWriteOptions("standard output"));
    }
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return GetImpl()->Write(strm, opts);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  explicit LinearTaggerFst(std::shared_ptr<Impl> impl)
      : ImplToFst<Impl>(impl) {}

  void operator=(const LinearTaggerFst<A> &fst) = delete;
};

// Specialization for LinearTaggerFst.
template <class Arc>
class StateIterator<LinearTaggerFst<Arc>>
    : public CacheStateIterator<LinearTaggerFst<Arc>> {
 public:
  explicit StateIterator(const LinearTaggerFst<Arc> &fst)
      : CacheStateIterator<LinearTaggerFst<Arc>>(fst, fst.GetMutableImpl()) {}
};

// Specialization for LinearTaggerFst.
template <class Arc>
class ArcIterator<LinearTaggerFst<Arc>>
    : public CacheArcIterator<LinearTaggerFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const LinearTaggerFst<Arc> &fst, StateId s)
      : CacheArcIterator<LinearTaggerFst<Arc>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class Arc>
inline void LinearTaggerFst<Arc>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<LinearTaggerFst<Arc>>(*this);
}

namespace internal {

// Implementation class for on-the-fly generated LinearClassifierFst with
// special optimization in matching.
template <class A>
class LinearClassifierFstImpl : public CacheImpl<A> {
 public:
  using FstImpl<A>::SetType;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;
  using FstImpl<A>::WriteHeader;

  using CacheBaseImpl<CacheState<A>>::PushArc;
  using CacheBaseImpl<CacheState<A>>::HasArcs;
  using CacheBaseImpl<CacheState<A>>::HasFinal;
  using CacheBaseImpl<CacheState<A>>::HasStart;
  using CacheBaseImpl<CacheState<A>>::SetArcs;
  using CacheBaseImpl<CacheState<A>>::SetFinal;
  using CacheBaseImpl<CacheState<A>>::SetStart;

  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef typename Collection<StateId, Label>::SetIterator NGramIterator;

  // Constructs an empty FST by default.
  LinearClassifierFstImpl()
      : CacheImpl<A>(CacheOptions()),
        data_(std::make_shared<LinearFstData<A>>()) {
    SetType("linear-classifier");
    num_classes_ = 0;
    num_groups_ = 0;
  }

  // Constructs the FST with given data storage, number of classes and
  // symbol tables.
  LinearClassifierFstImpl(const LinearFstData<Arc> *data, size_t num_classes,
                          const SymbolTable *isyms, const SymbolTable *osyms,
                          CacheOptions opts)
      : CacheImpl<A>(opts),
        data_(data),
        num_classes_(num_classes),
        num_groups_(data_->NumGroups() / num_classes_) {
    SetType("linear-classifier");
    SetProperties(kILabelSorted, kFstProperties);
    SetInputSymbols(isyms);
    SetOutputSymbols(osyms);
    ReserveStubSpace();
  }

  // Copy by sharing the underlying data storage.
  LinearClassifierFstImpl(const LinearClassifierFstImpl &impl)
      : CacheImpl<A>(impl),
        data_(impl.data_),
        num_classes_(impl.num_classes_),
        num_groups_(impl.num_groups_) {
    SetType("linear-classifier");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
    ReserveStubSpace();
  }

  StateId Start() {
    if (!HasStart()) {
      StateId start = FindStartState();
      SetStart(start);
    }
    return CacheImpl<A>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      state_stub_.clear();
      FillState(s, &state_stub_);
      SetFinal(s, FinalWeight(state_stub_));
    }
    return CacheImpl<A>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<A>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<A>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<A>::NumOutputEpsilons(s);
  }

  void InitArcIterator(StateId s, ArcIteratorData<A> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl<A>::InitArcIterator(s, data);
  }

  // Computes the outgoing transitions from a state, creating new
  // destination states as needed.
  void Expand(StateId s);

  // Appends to `arcs` all out-going arcs from state `s` that matches
  // `label` as the input label.
  void MatchInput(StateId s, Label ilabel, std::vector<Arc> *arcs);

  static LinearClassifierFstImpl<A> *Read(std::istream &strm,
                                          const FstReadOptions &opts);

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const {
    FstHeader header;
    header.SetStart(kNoStateId);
    WriteHeader(strm, opts, kFileVersion, &header);
    data_->Write(strm);
    WriteType(strm, num_classes_);
    if (!strm) {
      LOG(ERROR) << "LinearClassifierFst::Write: Write failed: " << opts.source;
      return false;
    }
    return true;
  }

 private:
  static const int kMinFileVersion;
  static const int kFileVersion;

  // A collection of functions to access parts of the state tuple. A
  // state tuple is a vector of `Label`s with two parts:
  //   [prediction] [internal].
  //
  // - [prediction] is a single label of the predicted class. A state
  //   must have a positive class label, unless it is the start state.
  //
  // - [internal] is the internal state tuple for `LinearFstData` of
  //   the given class; or kNoTrieNodeId's if in start state.
  Label &Prediction(std::vector<Label> &state) { return state[0]; }  // NOLINT
  Label Prediction(const std::vector<Label> &state) const { return state[0]; }

  Label &InternalAt(std::vector<Label> &state, int index) {  // NOLINT
    return state[index + 1];
  }
  Label InternalAt(const std::vector<Label> &state, int index) const {
    return state[index + 1];
  }

  // The size of state tuples are fixed, reserve them in stubs
  void ReserveStubSpace() {
    size_t size = 1 + num_groups_;
    state_stub_.reserve(size);
    next_stub_.reserve(size);
  }

  // Computes the start state tuple and maps it to the start state id.
  StateId FindStartState() {
    // A start state tuple has no prediction
    state_stub_.clear();
    state_stub_.push_back(kNoLabel);
    // For a start state, we don't yet know where we are in the tries.
    for (size_t i = 0; i < num_groups_; ++i)
      state_stub_.push_back(kNoTrieNodeId);
    return FindState(state_stub_);
  }

  // Tests if the state tuple represents the start state.
  bool IsStartState(const std::vector<Label> &state) const {
    return state[0] == kNoLabel;
  }

  // Computes the actual group id in the data storage.
  int GroupId(Label pred, int group) const {
    return group * num_classes_ + pred - 1;
  }

  // Finds out the final weight of the given state. A state is final
  // iff it is not the start.
  Weight FinalWeight(const std::vector<Label> &state) const {
    if (IsStartState(state)) {
      return Weight::Zero();
    }
    Label pred = Prediction(state);
    DCHECK_GT(pred, 0);
    DCHECK_LE(pred, num_classes_);
    Weight final_weight = Weight::One();
    for (size_t group = 0; group < num_groups_; ++group) {
      int group_id = GroupId(pred, group);
      int trie_state = InternalAt(state, group);
      final_weight =
          Times(final_weight, data_->GroupFinalWeight(group_id, trie_state));
    }
    return final_weight;
  }

  // Finds state corresponding to an n-gram. Creates new state if n-gram not
  // found.
  StateId FindState(const std::vector<Label> &ngram) {
    StateId sparse = ngrams_.FindId(ngram, true);
    StateId dense = condensed_.FindId(sparse, true);
    return dense;
  }

  // Appends after `output` the state tuple corresponding to the state id. The
  // state id must exist.
  void FillState(StateId s, std::vector<Label> *output) {
    s = condensed_.FindEntry(s);
    for (NGramIterator it = ngrams_.FindSet(s); !it.Done(); it.Next()) {
      Label label = it.Element();
      output->push_back(label);
    }
  }

  std::shared_ptr<const LinearFstData<A>> data_;
  // Division of groups in `data_`; num_classes_ * num_groups_ ==
  // data_->NumGroups().
  size_t num_classes_, num_groups_;
  // Mapping from internal state tuple to *non-consecutive* ids
  Collection<StateId, Label> ngrams_;
  // Mapping from non-consecutive id to actual state id
  CompactHashBiTable<StateId, StateId, std::hash<StateId>> condensed_;
  // Two frequently used vectors, reuse to avoid repeated heap
  // allocation
  std::vector<Label> state_stub_, next_stub_;

  void operator=(const LinearClassifierFstImpl<A> &) = delete;
};

template <class A>
const int LinearClassifierFstImpl<A>::kMinFileVersion = 0;

template <class A>
const int LinearClassifierFstImpl<A>::kFileVersion = 0;

template <class A>
void LinearClassifierFstImpl<A>::Expand(StateId s) {
  VLOG(3) << "Expand " << s;
  state_stub_.clear();
  FillState(s, &state_stub_);
  next_stub_.clear();
  next_stub_.resize(1 + num_groups_);

  if (IsStartState(state_stub_)) {
    // Make prediction
    for (Label pred = 1; pred <= num_classes_; ++pred) {
      Prediction(next_stub_) = pred;
      for (int i = 0; i < num_groups_; ++i)
        InternalAt(next_stub_, i) = data_->GroupStartState(GroupId(pred, i));
      PushArc(s, A(0, pred, Weight::One(), FindState(next_stub_)));
    }
  } else {
    Label pred = Prediction(state_stub_);
    DCHECK_GT(pred, 0);
    DCHECK_LE(pred, num_classes_);
    for (Label ilabel = data_->MinInputLabel();
         ilabel <= data_->MaxInputLabel(); ++ilabel) {
      Prediction(next_stub_) = pred;
      Weight weight = Weight::One();
      for (int i = 0; i < num_groups_; ++i)
        InternalAt(next_stub_, i) =
            data_->GroupTransition(GroupId(pred, i), InternalAt(state_stub_, i),
                                   ilabel, pred, &weight);
      PushArc(s, A(ilabel, 0, weight, FindState(next_stub_)));
    }
  }

  SetArcs(s);
}

template <class A>
void LinearClassifierFstImpl<A>::MatchInput(StateId s, Label ilabel,
                                            std::vector<Arc> *arcs) {
  state_stub_.clear();
  FillState(s, &state_stub_);
  next_stub_.clear();
  next_stub_.resize(1 + num_groups_);

  if (IsStartState(state_stub_)) {
    // Make prediction if `ilabel` is epsilon.
    if (ilabel == 0) {
      for (Label pred = 1; pred <= num_classes_; ++pred) {
        Prediction(next_stub_) = pred;
        for (int i = 0; i < num_groups_; ++i)
          InternalAt(next_stub_, i) = data_->GroupStartState(GroupId(pred, i));
        arcs->push_back(A(0, pred, Weight::One(), FindState(next_stub_)));
      }
    }
  } else if (ilabel != 0) {
    Label pred = Prediction(state_stub_);
    Weight weight = Weight::One();
    Prediction(next_stub_) = pred;
    for (int i = 0; i < num_groups_; ++i)
      InternalAt(next_stub_, i) = data_->GroupTransition(
          GroupId(pred, i), InternalAt(state_stub_, i), ilabel, pred, &weight);
    arcs->push_back(A(ilabel, 0, weight, FindState(next_stub_)));
  }
}

template <class A>
inline LinearClassifierFstImpl<A> *LinearClassifierFstImpl<A>::Read(
    std::istream &strm, const FstReadOptions &opts) {
  std::unique_ptr<LinearClassifierFstImpl<A>> impl(
      new LinearClassifierFstImpl<A>());
  FstHeader header;
  if (!impl->ReadHeader(strm, opts, kMinFileVersion, &header)) {
    return nullptr;
  }
  impl->data_ = std::shared_ptr<LinearFstData<A>>(LinearFstData<A>::Read(strm));
  if (!impl->data_) {
    return nullptr;
  }
  ReadType(strm, &impl->num_classes_);
  if (!strm) {
    return nullptr;
  }
  impl->num_groups_ = impl->data_->NumGroups() / impl->num_classes_;
  if (impl->num_groups_ * impl->num_classes_ != impl->data_->NumGroups()) {
    FSTERROR() << "Total number of feature groups is not a multiple of the "
                  "number of classes: num groups = "
               << impl->data_->NumGroups()
               << ", num classes = " << impl->num_classes_;
    return nullptr;
  }
  impl->ReserveStubSpace();
  return impl.release();
}

}  // namespace internal

// This class attaches interface to implementation and handles
// reference counting, delegating most methods to ImplToFst.
template <class A>
class LinearClassifierFst
    : public ImplToFst<internal::LinearClassifierFstImpl<A>> {
 public:
  friend class ArcIterator<LinearClassifierFst<A>>;
  friend class StateIterator<LinearClassifierFst<A>>;
  friend class LinearFstMatcherTpl<LinearClassifierFst<A>>;

  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef DefaultCacheStore<A> Store;
  typedef typename Store::State State;
  using Impl = internal::LinearClassifierFstImpl<A>;

  LinearClassifierFst() : ImplToFst<Impl>(std::make_shared<Impl>()) {}

  explicit LinearClassifierFst(LinearFstData<A> *data, size_t num_classes,
                               const SymbolTable *isyms = nullptr,
                               const SymbolTable *osyms = nullptr,
                               CacheOptions opts = CacheOptions())
      : ImplToFst<Impl>(
            std::make_shared<Impl>(data, num_classes, isyms, osyms, opts)) {}

  explicit LinearClassifierFst(const Fst<A> &fst)
      : ImplToFst<Impl>(std::make_shared<Impl>()) {
    LOG(FATAL) << "LinearClassifierFst: no constructor from arbitrary FST.";
  }

  // See Fst<>::Copy() for doc.
  LinearClassifierFst(const LinearClassifierFst<A> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this LinearClassifierFst. See Fst<>::Copy() for further doc.
  LinearClassifierFst<A> *Copy(bool safe = false) const override {
    return new LinearClassifierFst<A>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<A> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<A> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  MatcherBase<A> *InitMatcher(MatchType match_type) const override {
    return new LinearFstMatcherTpl<LinearClassifierFst<A>>(this, match_type);
  }

  static LinearClassifierFst<A> *Read(const string &filename) {
    if (!filename.empty()) {
      std::ifstream strm(filename,
                              std::ios_base::in | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "LinearClassifierFst::Read: Can't open file: "
                   << filename;
        return nullptr;
      }
      return Read(strm, FstReadOptions(filename));
    } else {
      return Read(std::cin, FstReadOptions("standard input"));
    }
  }

  static LinearClassifierFst<A> *Read(std::istream &in,
                                      const FstReadOptions &opts) {
    auto *impl = Impl::Read(in, opts);
    return impl ? new LinearClassifierFst<A>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  bool Write(const string &filename) const override {
    if (!filename.empty()) {
      std::ofstream strm(filename,
                               std::ios_base::out | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "ProdLmFst::Write: Can't open file: " << filename;
        return false;
      }
      return Write(strm, FstWriteOptions(filename));
    } else {
      return Write(std::cout, FstWriteOptions("standard output"));
    }
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return GetImpl()->Write(strm, opts);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  explicit LinearClassifierFst(std::shared_ptr<Impl> impl)
      : ImplToFst<Impl>(impl) {}

  void operator=(const LinearClassifierFst<A> &fst) = delete;
};

// Specialization for LinearClassifierFst.
template <class Arc>
class StateIterator<LinearClassifierFst<Arc>>
    : public CacheStateIterator<LinearClassifierFst<Arc>> {
 public:
  explicit StateIterator(const LinearClassifierFst<Arc> &fst)
      : CacheStateIterator<LinearClassifierFst<Arc>>(fst,
                                                     fst.GetMutableImpl()) {}
};

// Specialization for LinearClassifierFst.
template <class Arc>
class ArcIterator<LinearClassifierFst<Arc>>
    : public CacheArcIterator<LinearClassifierFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const LinearClassifierFst<Arc> &fst, StateId s)
      : CacheArcIterator<LinearClassifierFst<Arc>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class Arc>
inline void LinearClassifierFst<Arc>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<LinearClassifierFst<Arc>>(*this);
}

// Specialized Matcher for LinearFsts. This matcher only supports
// matching from the input side. This is intentional because comparing
// the scores of different input sequences with the same output
// sequence is meaningless in a discriminative model.
template <class F>
class LinearFstMatcherTpl : public MatcherBase<typename F::Arc> {
 public:
  typedef typename F::Arc Arc;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef F FST;

  // This makes a copy of the FST.
  LinearFstMatcherTpl(const FST &fst, MatchType match_type)
      : owned_fst_(fst.Copy()),
        fst_(*owned_fst_),
        match_type_(match_type),
        s_(kNoStateId),
        current_loop_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId),
        cur_arc_(0),
        error_(false) {
    switch (match_type_) {
      case MATCH_INPUT:
      case MATCH_OUTPUT:
      case MATCH_NONE:
        break;
      default:
        FSTERROR() << "LinearFstMatcherTpl: Bad match type";
        match_type_ = MATCH_NONE;
        error_ = true;
    }
  }

  // This doesn't copy the FST.
  LinearFstMatcherTpl(const FST *fst, MatchType match_type)
      : fst_(*fst),
        match_type_(match_type),
        s_(kNoStateId),
        current_loop_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId),
        cur_arc_(0),
        error_(false) {
    switch (match_type_) {
      case MATCH_INPUT:
      case MATCH_OUTPUT:
      case MATCH_NONE:
        break;
      default:
        FSTERROR() << "LinearFstMatcherTpl: Bad match type";
        match_type_ = MATCH_NONE;
        error_ = true;
    }
  }

  // This makes a copy of the FST.
  LinearFstMatcherTpl(const LinearFstMatcherTpl<F> &matcher, bool safe = false)
      : owned_fst_(matcher.fst_.Copy(safe)),
        fst_(*owned_fst_),
        match_type_(matcher.match_type_),
        s_(kNoStateId),
        current_loop_(false),
        loop_(matcher.loop_),
        cur_arc_(0),
        error_(matcher.error_) {}

  LinearFstMatcherTpl<F> *Copy(bool safe = false) const override {
    return new LinearFstMatcherTpl<F>(*this, safe);
  }

  MatchType Type(bool /*test*/) const override {
    // `MATCH_INPUT` is the only valid type
    return match_type_ == MATCH_INPUT ? match_type_ : MATCH_NONE;
  }

  void SetState(StateId s) final {
    if (s_ == s) return;
    s_ = s;
    // `MATCH_INPUT` is the only valid type
    if (match_type_ != MATCH_INPUT) {
      FSTERROR() << "LinearFstMatcherTpl: Bad match type";
      error_ = true;
    }
    loop_.nextstate = s;
  }

  bool Find(Label label) final {
    if (error_) {
      current_loop_ = false;
      return false;
    }
    current_loop_ = label == 0;
    if (label == kNoLabel) label = 0;
    arcs_.clear();
    cur_arc_ = 0;
    fst_.GetMutableImpl()->MatchInput(s_, label, &arcs_);
    return current_loop_ || !arcs_.empty();
  }

  bool Done() const final {
    return !(current_loop_ || cur_arc_ < arcs_.size());
  }

  const Arc &Value() const final {
    return current_loop_ ? loop_ : arcs_[cur_arc_];
  }

  void Next() final {
    if (current_loop_)
      current_loop_ = false;
    else
      ++cur_arc_;
  }

  std::ptrdiff_t Priority(StateId s) final { return kRequirePriority; }

  const FST &GetFst() const override { return fst_; }

  uint64 Properties(uint64 props) const override {
    if (error_) props |= kError;
    return props;
  }

  uint32 Flags() const override { return kRequireMatch; }

 private:
  std::unique_ptr<const FST> owned_fst_;
  const FST &fst_;
  MatchType match_type_;  // Type of match to perform.
  StateId s_;             // Current state.
  bool current_loop_;     // Current arc is the implicit loop.
  Arc loop_;              // For non-consuming symbols.
  // All out-going arcs matching the label in last Find() call.
  std::vector<Arc> arcs_;
  size_t cur_arc_;  // Index to the arc that `Value()` should return.
  bool error_;      // Error encountered.
};

}  // namespace fst

#endif  // FST_EXTENSIONS_LINEAR_LINEAR_FST_H_
