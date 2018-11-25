// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// FST Class for memory-efficient representation of common types of
// FSTs: linear automata, acceptors, unweighted FSTs, ...

#ifndef FST_COMPACT_FST_H_
#define FST_COMPACT_FST_H_

#include <climits>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/expanded-fst.h>
#include <fst/fst-decl.h>  // For optional argument declarations
#include <fst/mapped-file.h>
#include <fst/matcher.h>
#include <fst/test-properties.h>
#include <fst/util.h>


namespace fst {

struct CompactFstOptions : public CacheOptions {
  // The default caching behaviour is to do no caching. Most compactors are
  // cheap and therefore we save memory by not doing caching.
  CompactFstOptions() : CacheOptions(true, 0) {}

  explicit CompactFstOptions(const CacheOptions &opts) : CacheOptions(opts) {}
};

// New upcoming (Fst) Compactor interface - currently used internally
// by CompactFstImpl.
//
// class Compactor {
//  public:
//   // Constructor from the Fst to be compacted.
//   Compactor(const Fst<Arc> &fst, ...);
//   // Copy constructor
//   Compactor(const Compactor &compactor, bool safe = false)
//   // Default constructor (optional, see comment below).
//   Compactor();
//
//   // Returns the start state, number of states, and total number of arcs
//   // of the compacted Fst
//   StateId Start() const;
//   StateId NumStates() const;
//   size_t NumArcs() const;
//
//   // Accessor class for state attributes.
//   class State {
//    public:
//     State();  // Required, corresponds to kNoStateId.
//     State(const Compactor *c, StateId);  // Accessor for StateId 's'.
//     StateId GetStateId() const;
//     Weight Final() const;
//     size_t NumArcs() const;
//     Arc GetArc(size_t i) const;
//   };
//
//   // Modifies 'state' accessor to provide access to state id 's'.
//   void SetState(StateId s, State *state);
//   // Tests whether 'fst' can be compacted by this compactor.
//   bool IsCompatible(const Fst<A> &fst) const;
//   // Return the properties that are always true for an fst
//   // compacted using this compactor
//   uint64 Properties() const;
//   // Return a string identifying the type of compactor.
//   static const string &Type();
//   // Return true if an error has occured.
//   bool Error() const;
//   // Writes a compactor to a file.
//   bool Write(std::ostream &strm, const FstWriteOptions &opts) const;
//   // Reads a compactor from a file.
//   static Compactor*Read(std::istream &strm, const FstReadOptions &opts,
//                         const FstHeader &hdr);
// };
//

// Old (Arc) Compactor Interface:
//
// The ArcCompactor class determines how arcs and final weights are compacted
// and expanded.
//
// Final weights are treated as transitions to the superfinal state, i.e.,
// ilabel = olabel = kNoLabel and nextstate = kNoStateId.
//
// There are two types of compactors:
//
// * Fixed out-degree compactors: 'compactor.Size()' returns a positive integer
//   's'. An FST can be compacted by this compactor only if each state has
//   exactly 's' outgoing transitions (counting a non-Zero() final weight as a
//   transition). A typical example is a compactor for string FSTs, i.e.,
//   's == 1'.
//
// * Variable out-degree compactors: 'compactor.Size() == -1'. There are no
//   out-degree restrictions for these compactors.
//
// Interface:
//
// class ArcCompactor {
//  public:
//   // Element is the type of the compacted transitions.
//   using Element = ...
//
//   // Returns the compacted representation of a transition 'arc'
//   // at a state 's'.
//   Element Compact(StateId s, const Arc &arc);
//
//   // Returns the transition at state 's' represented by the compacted
//   // transition 'e'.
//   Arc Expand(StateId s, const Element &e) const;
//
//   // Returns -1 for variable out-degree compactors, and the mandatory
//   // out-degree otherwise.
//   ssize_t Size() const;
//
//   // Tests whether an FST can be compacted by this compactor.
//   bool Compatible(const Fst<A> &fst) const;
//
//   // Returns the properties that are always true for an FST compacted using
//   // this compactor
//   uint64 Properties() const;
//
//   // Returns a string identifying the type of compactor.
//   static const string &Type();
//
//   // Writes a compactor to a file.
//   bool Write(std::ostream &strm) const;
//
//   // Reads a compactor from a file.
//   static ArcCompactor *Read(std::istream &strm);
//
//   // Default constructor (optional, see comment below).
//   ArcCompactor();
// };
//
// The default constructor is only required for FST_REGISTER to work (i.e.,
// enabling Convert() and the command-line utilities to work with this new
// compactor). However, a default constructor always needs to be specified for
// this code to compile, but one can have it simply raise an error when called,
// like so:
//
// Compactor::Compactor() {
//   FSTERROR() << "Compactor: No default constructor";
// }

// Default implementation data for CompactFst, which can shared between
// otherwise independent copies.
//
// The implementation contains two arrays: 'states_' and 'compacts_'.
//
// For fixed out-degree compactors, the 'states_' array is unallocated. The
// 'compacts_' contains the compacted transitions. Its size is 'ncompacts_'.
// The outgoing transitions at a given state are stored consecutively. For a
// given state 's', its 'compactor.Size()' outgoing transitions (including
// superfinal transition when 's' is final), are stored in position
// ['s*compactor.Size()', '(s+1)*compactor.Size()').
//
// For variable out-degree compactors, the states_ array has size
// 'nstates_ + 1' and contains pointers to positions into 'compacts_'. For a
// given state 's', the compacted transitions of 's' are stored in positions
// ['states_[s]', 'states_[s + 1]') in 'compacts_'. By convention,
// 'states_[nstates_] == ncompacts_'.
//
// In both cases, the superfinal transitions (when 's' is final, i.e.,
// 'Final(s) != Weight::Zero()') are stored first.
//
// The unsigned type U is used to represent indices into the compacts_ array.
template <class Element, class Unsigned>
class DefaultCompactStore {
 public:
  DefaultCompactStore()
      : states_(nullptr),
        compacts_(nullptr),
        nstates_(0),
        ncompacts_(0),
        narcs_(0),
        start_(kNoStateId),
        error_(false) {}

  template <class Arc, class Compactor>
  DefaultCompactStore(const Fst<Arc> &fst, const Compactor &compactor);

  template <class Iterator, class Compactor>
  DefaultCompactStore(const Iterator &begin, const Iterator &end,
                      const Compactor &compactor);

  ~DefaultCompactStore() {
    if (!states_region_) delete[] states_;
    if (!compacts_region_) delete[] compacts_;
  }

  template <class Compactor>
  static DefaultCompactStore<Element, Unsigned> *Read(
      std::istream &strm, const FstReadOptions &opts, const FstHeader &hdr,
      const Compactor &compactor);

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const;

  Unsigned States(ssize_t i) const { return states_[i]; }

  const Element &Compacts(size_t i) const { return compacts_[i]; }

  size_t NumStates() const { return nstates_; }

  size_t NumCompacts() const { return ncompacts_; }

  size_t NumArcs() const { return narcs_; }

  ssize_t Start() const { return start_; }

  bool Error() const { return error_; }

  // Returns a string identifying the type of data storage container.
  static const string &Type();

 private:
  std::unique_ptr<MappedFile> states_region_;
  std::unique_ptr<MappedFile> compacts_region_;
  Unsigned *states_;
  Element *compacts_;
  size_t nstates_;
  size_t ncompacts_;
  size_t narcs_;
  ssize_t start_;
  bool error_;
};

template <class Element, class Unsigned>
template <class Arc, class Compactor>
DefaultCompactStore<Element, Unsigned>::DefaultCompactStore(
    const Fst<Arc> &fst, const Compactor &compactor)
    : states_(nullptr),
      compacts_(nullptr),
      nstates_(0),
      ncompacts_(0),
      narcs_(0),
      start_(kNoStateId),
      error_(false) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  start_ = fst.Start();
  // Counts # of states and arcs.
  StateId nfinals = 0;
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    ++nstates_;
    const auto s = siter.Value();
    for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      ++narcs_;
    }
    if (fst.Final(s) != Weight::Zero()) ++nfinals;
  }
  if (compactor.Size() == -1) {
    states_ = new Unsigned[nstates_ + 1];
    ncompacts_ = narcs_ + nfinals;
    compacts_ = new Element[ncompacts_];
    states_[nstates_] = ncompacts_;
  } else {
    states_ = nullptr;
    ncompacts_ = nstates_ * compactor.Size();
    if ((narcs_ + nfinals) != ncompacts_) {
      FSTERROR() << "DefaultCompactStore: Compactor incompatible with FST";
      error_ = true;
      return;
    }
    compacts_ = new Element[ncompacts_];
  }
  size_t pos = 0;
  size_t fpos = 0;
  for (size_t s = 0; s < nstates_; ++s) {
    fpos = pos;
    if (compactor.Size() == -1) states_[s] = pos;
    if (fst.Final(s) != Weight::Zero()) {
      compacts_[pos++] = compactor.Compact(
          s, Arc(kNoLabel, kNoLabel, fst.Final(s), kNoStateId));
    }
    for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      compacts_[pos++] = compactor.Compact(s, aiter.Value());
    }
    if ((compactor.Size() != -1) && ((pos - fpos) != compactor.Size())) {
      FSTERROR() << "DefaultCompactStore: Compactor incompatible with FST";
      error_ = true;
      return;
    }
  }
  if (pos != ncompacts_) {
    FSTERROR() << "DefaultCompactStore: Compactor incompatible with FST";
    error_ = true;
    return;
  }
}

template <class Element, class Unsigned>
template <class Iterator, class Compactor>
DefaultCompactStore<Element, Unsigned>::DefaultCompactStore(
    const Iterator &begin, const Iterator &end, const Compactor &compactor)
    : states_(nullptr),
      compacts_(nullptr),
      nstates_(0),
      ncompacts_(0),
      narcs_(0),
      start_(kNoStateId),
      error_(false) {
  using Arc = typename Compactor::Arc;
  using Weight = typename Arc::Weight;
  if (compactor.Size() != -1) {
    ncompacts_ = std::distance(begin, end);
    if (compactor.Size() == 1) {
      // For strings, allows implicit final weight. Empty input is the empty
      // string.
      if (ncompacts_ == 0) {
        ++ncompacts_;
      } else {
        const auto arc =
            compactor.Expand(ncompacts_ - 1, *(begin + (ncompacts_ - 1)));
        if (arc.ilabel != kNoLabel) ++ncompacts_;
      }
    }
    if (ncompacts_ % compactor.Size()) {
      FSTERROR() << "DefaultCompactStore: Size of input container incompatible"
                 << " with compactor";
      error_ = true;
      return;
    }
    if (ncompacts_ == 0) return;
    start_ = 0;
    nstates_ = ncompacts_ / compactor.Size();
    compacts_ = new Element[ncompacts_];
    size_t i = 0;
    Iterator it = begin;
    for (; it != end; ++it, ++i) {
      compacts_[i] = *it;
      if (compactor.Expand(i, *it).ilabel != kNoLabel) ++narcs_;
    }
    if (i < ncompacts_) {
      compacts_[i] = compactor.Compact(
          i, Arc(kNoLabel, kNoLabel, Weight::One(), kNoStateId));
    }
  } else {
    if (std::distance(begin, end) == 0) return;
    // Count # of states, arcs and compacts.
    auto it = begin;
    for (size_t i = 0; it != end; ++it, ++i) {
      const auto arc = compactor.Expand(i, *it);
      if (arc.ilabel != kNoLabel) {
        ++narcs_;
        ++ncompacts_;
      } else {
        ++nstates_;
        if (arc.weight != Weight::Zero()) ++ncompacts_;
      }
    }
    start_ = 0;
    compacts_ = new Element[ncompacts_];
    states_ = new Unsigned[nstates_ + 1];
    states_[nstates_] = ncompacts_;
    size_t i = 0;
    size_t s = 0;
    for (it = begin; it != end; ++it) {
      const auto arc = compactor.Expand(i, *it);
      if (arc.ilabel != kNoLabel) {
        compacts_[i++] = *it;
      } else {
        states_[s++] = i;
        if (arc.weight != Weight::Zero()) compacts_[i++] = *it;
      }
    }
    if ((s != nstates_) || (i != ncompacts_)) {
      FSTERROR() << "DefaultCompactStore: Ill-formed input container";
      error_ = true;
      return;
    }
  }
}

template <class Element, class Unsigned>
template <class Compactor>
DefaultCompactStore<Element, Unsigned>
    *DefaultCompactStore<Element, Unsigned>::Read(std::istream &strm,
                                                  const FstReadOptions &opts,
                                                  const FstHeader &hdr,
                                                  const Compactor &compactor) {
  std::unique_ptr<DefaultCompactStore<Element, Unsigned>> data(
      new DefaultCompactStore<Element, Unsigned>());
  data->start_ = hdr.Start();
  data->nstates_ = hdr.NumStates();
  data->narcs_ = hdr.NumArcs();
  if (compactor.Size() == -1) {
    if ((hdr.GetFlags() & FstHeader::IS_ALIGNED) && !AlignInput(strm)) {
      LOG(ERROR) << "DefaultCompactStore::Read: Alignment failed: "
                 << opts.source;
      return nullptr;
    }
    auto b = (data->nstates_ + 1) * sizeof(Unsigned);
    data->states_region_.reset(MappedFile::Map(
        &strm, opts.mode == FstReadOptions::MAP, opts.source, b));
    if (!strm || !data->states_region_) {
      LOG(ERROR) << "DefaultCompactStore::Read: Read failed: " << opts.source;
      return nullptr;
    }
    data->states_ =
        static_cast<Unsigned *>(data->states_region_->mutable_data());
  } else {
    data->states_ = nullptr;
  }
  data->ncompacts_ = compactor.Size() == -1 ? data->states_[data->nstates_]
                                            : data->nstates_ * compactor.Size();
  if ((hdr.GetFlags() & FstHeader::IS_ALIGNED) && !AlignInput(strm)) {
    LOG(ERROR) << "DefaultCompactStore::Read: Alignment failed: "
               << opts.source;
    return nullptr;
  }
  size_t b = data->ncompacts_ * sizeof(Element);
  data->compacts_region_.reset(
      MappedFile::Map(&strm, opts.mode == FstReadOptions::MAP, opts.source, b));
  if (!strm || !data->compacts_region_) {
    LOG(ERROR) << "DefaultCompactStore::Read: Read failed: " << opts.source;
    return nullptr;
  }
  data->compacts_ =
      static_cast<Element *>(data->compacts_region_->mutable_data());
  return data.release();
}

template <class Element, class Unsigned>
bool DefaultCompactStore<Element, Unsigned>::Write(
    std::ostream &strm, const FstWriteOptions &opts) const {
  if (states_) {
    if (opts.align && !AlignOutput(strm)) {
      LOG(ERROR) << "DefaultCompactStore::Write: Alignment failed: "
                 << opts.source;
      return false;
    }
    strm.write(reinterpret_cast<char *>(states_),
               (nstates_ + 1) * sizeof(Unsigned));
  }
  if (opts.align && !AlignOutput(strm)) {
    LOG(ERROR) << "DefaultCompactStore::Write: Alignment failed: "
               << opts.source;
    return false;
  }
  strm.write(reinterpret_cast<char *>(compacts_), ncompacts_ * sizeof(Element));
  strm.flush();
  if (!strm) {
    LOG(ERROR) << "DefaultCompactStore::Write: Write failed: " << opts.source;
    return false;
  }
  return true;
}

template <class Element, class Unsigned>
const string &DefaultCompactStore<Element, Unsigned>::Type() {
  static const string *const type = new string("compact");
  return *type;
}

template <class C, class U, class S> class DefaultCompactState;

// Wraps an arc compactor and a compact store as a new Fst compactor.
template <class C, class U,
          class S = DefaultCompactStore<typename C::Element, U>>
class DefaultCompactor {
 public:
  using ArcCompactor = C;
  using Unsigned = U;
  using CompactStore = S;
  using Element = typename C::Element;
  using Arc = typename C::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using State = DefaultCompactState<C, U, S>;
  friend State;

  DefaultCompactor()
      : arc_compactor_(nullptr), compact_store_(nullptr) {}

  // Constructs from Fst.
  DefaultCompactor(const Fst<Arc> &fst,
                   std::shared_ptr<ArcCompactor> arc_compactor)
      : arc_compactor_(std::move(arc_compactor)),
        compact_store_(std::make_shared<S>(fst, *arc_compactor_)) {}

  DefaultCompactor(const Fst<Arc> &fst,
                   std::shared_ptr<DefaultCompactor<C, U, S>> compactor)
      : arc_compactor_(compactor->arc_compactor_),
        compact_store_(compactor->compact_store_ == nullptr ?
                       std::make_shared<S>(fst, *arc_compactor_) :
                       compactor->compact_store_) {}

  // Constructs from CompactStore.
  DefaultCompactor(std::shared_ptr<ArcCompactor> arc_compactor,
                   std::shared_ptr<CompactStore> compact_store)
      : arc_compactor_(std::move(arc_compactor)),
        compact_store_(std::move(compact_store)) {}

  // Constructs from set of compact elements (when arc_compactor.Size() != -1).
  template <class Iterator>
  DefaultCompactor(const Iterator &b, const Iterator &e,
                   std::shared_ptr<C> arc_compactor)
      : arc_compactor_(std::move(arc_compactor)),
        compact_store_(std::make_shared<S>(b, e, *arc_compactor_)) {}

  // Copy constructor.
  DefaultCompactor(const DefaultCompactor<C, U, S> &compactor)
      : arc_compactor_(std::make_shared<C>(*compactor.GetArcCompactor())),
        compact_store_(compactor.SharedCompactStore()) {}

  template <class OtherC>
  explicit DefaultCompactor(const DefaultCompactor<OtherC, U, S> &compactor)
      : arc_compactor_(std::make_shared<C>(*compactor.GetArcCompactor())),
        compact_store_(compactor.SharedCompactStore()) {}

  StateId Start() const { return compact_store_->Start(); }
  StateId NumStates() const { return compact_store_->NumStates(); }
  size_t NumArcs() const { return compact_store_->NumArcs(); }

  void SetState(StateId s, State *state) const {
    if (state->GetStateId() != s) state->Set(this, s);
  }

  static DefaultCompactor<C, U, S> *Read(std::istream &strm,
                                         const FstReadOptions &opts,
                                         const FstHeader &hdr) {
    std::shared_ptr<C> arc_compactor(C::Read(strm));
    if (arc_compactor == nullptr) return nullptr;
    std::shared_ptr<S> compact_store(S::Read(strm, opts, hdr, *arc_compactor));
    if (compact_store == nullptr) return nullptr;
    return new DefaultCompactor<C, U, S>(arc_compactor, compact_store);
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const {
    return arc_compactor_->Write(strm) && compact_store_->Write(strm, opts);
  }

  uint64 Properties() const { return arc_compactor_->Properties(); }

  bool IsCompatible(const Fst<Arc> &fst) const {
    return arc_compactor_->Compatible(fst);
  }

  bool Error() const { return compact_store_->Error(); }

  bool HasFixedOutdegree() const { return arc_compactor_->Size() != -1; }

  static const string &Type() {
    static const string *const type = [] {
      string type = "compact";
      if (sizeof(U) != sizeof(uint32)) type += std::to_string(8 * sizeof(U));
      type += "_";
      type += C::Type();
      if (CompactStore::Type() != "compact") {
        type += "_";
        type += CompactStore::Type();
      }
      return new string(type);
    }();
    return *type;
  }

  const ArcCompactor *GetArcCompactor() const { return arc_compactor_.get(); }
  CompactStore *GetCompactStore() const { return compact_store_.get(); }

  std::shared_ptr<ArcCompactor> SharedArcCompactor() const {
    return arc_compactor_;
  }

  std::shared_ptr<CompactStore> SharedCompactStore() const {
    return compact_store_;
  }

  // TODO(allauzen): remove dependencies on this method and make private.
  Arc ComputeArc(StateId s, Unsigned i, uint32 f) const {
    return arc_compactor_->Expand(s, compact_store_->Compacts(i), f);
  }

 private:
  std::pair<Unsigned, Unsigned> CompactsRange(StateId s) const {
    std::pair<size_t, size_t> range;
    if (HasFixedOutdegree()) {
      range.first = s * arc_compactor_->Size();
      range.second = arc_compactor_->Size();
    } else {
      range.first = compact_store_->States(s);
      range.second = compact_store_->States(s + 1) - range.first;
    }
    return range;
  }

 private:
  std::shared_ptr<ArcCompactor> arc_compactor_;
  std::shared_ptr<CompactStore> compact_store_;
};

// Default implementation of state attributes accessor class for
// DefaultCompactor. Use of efficient specialization strongly encouraged.
template <class C, class U, class S>
class DefaultCompactState {
 public:
  using Arc = typename C::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  DefaultCompactState() = default;

  DefaultCompactState(const DefaultCompactor<C, U, S> *compactor, StateId s)
      : compactor_(compactor),
        s_(s),
        range_(compactor->CompactsRange(s)),
        has_final_(
            range_.second != 0 &&
            compactor->ComputeArc(s, range_.first,
                                 kArcILabelValue).ilabel == kNoLabel) {
    if (has_final_) {
      ++range_.first;
      --range_.second;
    }
  }

  void Set(const DefaultCompactor<C, U, S> *compactor, StateId s) {
    compactor_ = compactor;
    s_ = s;
    range_ = compactor->CompactsRange(s);
    if (range_.second != 0 &&
        compactor->ComputeArc(s, range_.first, kArcILabelValue).ilabel
        == kNoLabel) {
      has_final_ = true;
      ++range_.first;
      --range_.second;
    } else {
      has_final_ = false;
    }
  }

  StateId GetStateId() const { return s_; }

  Weight Final() const {
    if (!has_final_) return Weight::Zero();
    return compactor_->ComputeArc(s_, range_.first - 1, kArcWeightValue).weight;
  }

  size_t NumArcs() const { return range_.second; }

  Arc GetArc(size_t i, uint32 f) const {
    return compactor_->ComputeArc(s_, range_.first + i, f);
  }

 private:
  const DefaultCompactor<C, U, S> *compactor_ = nullptr;  // borrowed ref.
  StateId s_ = kNoStateId;
  std::pair<U, U> range_ = {0, 0};
  bool has_final_ = false;
};

// Specialization for DefaultCompactStore.
template <class C, class U>
class DefaultCompactState<C, U, DefaultCompactStore<typename C::Element, U>> {
 public:
  using Arc = typename C::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using CompactStore = DefaultCompactStore<typename C::Element, U>;

  DefaultCompactState() = default;

  DefaultCompactState(
      const DefaultCompactor<C, U, CompactStore> *compactor, StateId s)
      : arc_compactor_(compactor->GetArcCompactor()), s_(s) {
    Init(compactor);
  }

  void Set(const DefaultCompactor<C, U, CompactStore> *compactor, StateId s) {
    arc_compactor_ = compactor->GetArcCompactor();
    s_ = s;
    has_final_ = false;
    Init(compactor);
  }

  StateId GetStateId() const { return s_; }

  Weight Final() const {
    if (!has_final_) return Weight::Zero();
    return arc_compactor_->Expand(s_, *(compacts_ - 1), kArcWeightValue).weight;
  }

  size_t NumArcs() const { return num_arcs_; }

  Arc GetArc(size_t i, uint32 f) const {
    return arc_compactor_->Expand(s_, compacts_[i], f);
  }

 private:
  void Init(const DefaultCompactor<C, U, CompactStore> *compactor) {
    const auto *store = compactor->GetCompactStore();
    U offset;
    if (!compactor->HasFixedOutdegree()) {  // Variable out-degree compactor.
      offset = store->States(s_);
      num_arcs_ = store->States(s_ + 1) - offset;
    } else {  // Fixed out-degree compactor.
      offset = s_ * arc_compactor_->Size();
      num_arcs_ = arc_compactor_->Size();
    }
    if (num_arcs_ > 0) {
      compacts_ = &(store->Compacts(offset));
      if (arc_compactor_->Expand(s_, *compacts_, kArcILabelValue).ilabel
          == kNoStateId) {
        ++compacts_;
        --num_arcs_;
        has_final_ = true;
      }
    }
  }

 private:
  const C *arc_compactor_ = nullptr;               // Borrowed reference.
  const typename C::Element *compacts_ = nullptr;  // Borrowed reference.
  StateId s_ = kNoStateId;
  U num_arcs_ = 0;
  bool has_final_ = false;
};

template <class Arc, class ArcCompactor, class Unsigned, class CompactStore,
          class CacheStore>
class CompactFst;

template <class F, class G>
void Cast(const F &, G *);

namespace internal {

// Implementation class for CompactFst, which contains parametrizeable
// Fst data storage (DefaultCompactStore by default) and Fst cache.
template <class Arc, class C, class CacheStore = DefaultCacheStore<Arc>>
class CompactFstImpl
    : public CacheBaseImpl<typename CacheStore::State, CacheStore> {
 public:
  using Weight = typename Arc::Weight;
  using StateId = typename Arc::StateId;
  using Compactor = C;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;
  using FstImpl<Arc>::WriteHeader;

  using ImplBase = CacheBaseImpl<typename CacheStore::State, CacheStore>;
  using ImplBase::PushArc;
  using ImplBase::HasArcs;
  using ImplBase::HasFinal;
  using ImplBase::HasStart;
  using ImplBase::SetArcs;
  using ImplBase::SetFinal;
  using ImplBase::SetStart;

  CompactFstImpl()
      : ImplBase(CompactFstOptions()),
        compactor_() {
    SetType(Compactor::Type());
    SetProperties(kNullProperties | kStaticProperties);
  }

  CompactFstImpl(const Fst<Arc> &fst, std::shared_ptr<Compactor> compactor,
                 const CompactFstOptions &opts)
      : ImplBase(opts),
        compactor_(std::make_shared<Compactor>(fst, compactor)) {
    SetType(Compactor::Type());
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
    if (compactor_->Error()) SetProperties(kError, kError);
    uint64 copy_properties = fst.Properties(kMutable, false) ?
        fst.Properties(kCopyProperties, true):
        CheckProperties(fst,
                        kCopyProperties & ~kWeightedCycles & ~kUnweightedCycles,
                        kCopyProperties);
    if ((copy_properties & kError) || !compactor_->IsCompatible(fst)) {
      FSTERROR() << "CompactFstImpl: Input Fst incompatible with compactor";
      SetProperties(kError, kError);
      return;
    }
    SetProperties(copy_properties | kStaticProperties);
  }

  CompactFstImpl(std::shared_ptr<Compactor> compactor,
                 const CompactFstOptions &opts)
      : ImplBase(opts),
        compactor_(compactor) {
    SetType(Compactor::Type());
    SetProperties(kStaticProperties | compactor_->Properties());
    if (compactor_->Error()) SetProperties(kError, kError);
  }

  CompactFstImpl(const CompactFstImpl<Arc, Compactor, CacheStore> &impl)
      : ImplBase(impl),
        compactor_(impl.compactor_ == nullptr ?
                   std::make_shared<Compactor>() :
                   std::make_shared<Compactor>(*impl.compactor_)) {
    SetType(impl.Type());
    SetProperties(impl.Properties());
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  // Allows to change the cache store from OtherI to I.
  template <class OtherCacheStore>
  CompactFstImpl(const CompactFstImpl<Arc, Compactor, OtherCacheStore> &impl)
      : ImplBase(CacheOptions(impl.GetCacheGc(), impl.GetCacheLimit())),
        compactor_(impl.compactor_ == nullptr ?
                   std::make_shared<Compactor>() :
                   std::make_shared<Compactor>(*impl.compactor_)) {
    SetType(impl.Type());
    SetProperties(impl.Properties());
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  StateId Start() {
    if (!HasStart()) SetStart(compactor_->Start());
    return ImplBase::Start();
  }

  Weight Final(StateId s) {
    if (HasFinal(s)) return ImplBase::Final(s);
    compactor_->SetState(s, &state_);
    return state_.Final();
  }

  StateId NumStates() const {
    if (Properties(kError)) return 0;
    return compactor_->NumStates();
  }

  size_t NumArcs(StateId s) {
    if (HasArcs(s)) return ImplBase::NumArcs(s);
    compactor_->SetState(s, &state_);
    return state_.NumArcs();
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s) && !Properties(kILabelSorted)) Expand(s);
    if (HasArcs(s)) return ImplBase::NumInputEpsilons(s);
    return CountEpsilons(s, false);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s) && !Properties(kOLabelSorted)) Expand(s);
    if (HasArcs(s)) return ImplBase::NumOutputEpsilons(s);
    return CountEpsilons(s, true);
  }

  size_t CountEpsilons(StateId s, bool output_epsilons) {
    compactor_->SetState(s, &state_);
    const uint32 f = output_epsilons ? kArcOLabelValue : kArcILabelValue;
    size_t num_eps = 0;
    for (size_t i = 0; i < state_.NumArcs(); ++i) {
      const auto& arc = state_.GetArc(i, f);
      const auto label = output_epsilons ? arc.olabel : arc.ilabel;
      if (label == 0)
        ++num_eps;
      else if (label > 0)
        break;
    }
    return num_eps;
  }

  static CompactFstImpl<Arc, Compactor, CacheStore> *Read(
      std::istream &strm, const FstReadOptions &opts) {
    std::unique_ptr<CompactFstImpl<Arc, Compactor, CacheStore>> impl(
      new CompactFstImpl<Arc, Compactor, CacheStore>());
    FstHeader hdr;
    if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) {
      return nullptr;
    }
    // Ensures compatibility.
    if (hdr.Version() == kAlignedFileVersion) {
      hdr.SetFlags(hdr.GetFlags() | FstHeader::IS_ALIGNED);
    }
    impl->compactor_ = std::shared_ptr<Compactor>(
        Compactor::Read(strm, opts, hdr));
    if (!impl->compactor_) {
      return nullptr;
    }
    return impl.release();
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const {
    FstHeader hdr;
    hdr.SetStart(compactor_->Start());
    hdr.SetNumStates(compactor_->NumStates());
    hdr.SetNumArcs(compactor_->NumArcs());
    // Ensures compatibility.
    const auto file_version = opts.align ? kAlignedFileVersion : kFileVersion;
    WriteHeader(strm, opts, file_version, &hdr);
    return compactor_->Write(strm, opts);
  }

  // Provides information needed for generic state iterator.
  void InitStateIterator(StateIteratorData<Arc> *data) const {
    data->base = nullptr;
    data->nstates = compactor_->NumStates();
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) Expand(s);
    ImplBase::InitArcIterator(s, data);
  }

  void Expand(StateId s) {
    compactor_->SetState(s, &state_);
    for (size_t i = 0; i < state_.NumArcs(); ++i)
      PushArc(s, state_.GetArc(i, kArcValueFlags));
    SetArcs(s);
    if (!HasFinal(s)) SetFinal(s, state_.Final());
  }

  const Compactor *GetCompactor() const { return compactor_.get(); }
  std::shared_ptr<Compactor> SharedCompactor() const { return compactor_; }
  void SetCompactor(std::shared_ptr<Compactor> compactor) {
    // TODO(allauzen): is this correct? is this needed?
    // TODO(allauzen): consider removing and forcing this through direct calls
    // to compactor.
    compactor_ = compactor;
  }

  // Properties always true of this FST class.
  static constexpr uint64 kStaticProperties = kExpanded;

 protected:
  template <class OtherArc, class OtherCompactor, class OtherCacheStore>
  explicit CompactFstImpl(
    const CompactFstImpl<OtherArc, OtherCompactor, OtherCacheStore> &impl)
    : compactor_(std::make_shared<Compactor>(*impl.GetCompactor())) {
    SetType(impl.Type());
    SetProperties(impl.Properties());
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

 private:
  // Allows access during write.
  template <class AnyArc, class ArcCompactor, class Unsigned,
            class CompactStore, class AnyCacheStore>
  friend class ::fst::CompactFst;  // allow access during write.

  // Current unaligned file format version.
  static constexpr int kFileVersion = 2;
  // Current aligned file format version.
  static constexpr int kAlignedFileVersion = 1;
  // Minimum file format version supported.
  static constexpr int kMinFileVersion = 1;

  std::shared_ptr<Compactor> compactor_;
  typename Compactor::State state_;
};

template <class Arc, class Compactor, class CacheStore>
constexpr uint64 CompactFstImpl<Arc, Compactor, CacheStore>::kStaticProperties;

template <class Arc, class Compactor, class CacheStore>
constexpr int CompactFstImpl<Arc, Compactor, CacheStore>::kFileVersion;

template <class Arc, class Compactor, class CacheStore>
constexpr int CompactFstImpl<Arc, Compactor, CacheStore>::kAlignedFileVersion;

template <class Arc, class Compactor, class CacheStore>
constexpr int CompactFstImpl<Arc, Compactor, CacheStore>::kMinFileVersion;

}  // namespace internal

// This class attaches interface to implementation and handles reference
// counting, delegating most methods to ImplToExpandedFst. The Unsigned type
// is used to represent indices into the compact arc array. (Template
// argument defaults are declared in fst-decl.h.)
template <class A, class ArcCompactor, class Unsigned, class CompactStore,
          class CacheStore>
class CompactFst
    : public ImplToExpandedFst<internal::CompactFstImpl<
          A,
          DefaultCompactor<ArcCompactor, Unsigned, CompactStore>,
          CacheStore>> {
 public:
  template <class F, class G>
  void friend Cast(const F &, G *);

  using Arc = A;
  using StateId = typename A::StateId;
  using Compactor = DefaultCompactor<ArcCompactor, Unsigned, CompactStore>;
  using Impl = internal::CompactFstImpl<A, Compactor, CacheStore>;
  using Store = CacheStore;  // for CacheArcIterator

  friend class StateIterator<
      CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>>;
  friend class ArcIterator<
      CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>>;

  CompactFst() : ImplToExpandedFst<Impl>(std::make_shared<Impl>()) {}

  // If data is not nullptr, it is assumed to be already initialized.
  explicit CompactFst(
      const Fst<A> &fst,
      const ArcCompactor &compactor = ArcCompactor(),
      const CompactFstOptions &opts = CompactFstOptions(),
      std::shared_ptr<CompactStore> data = std::shared_ptr<CompactStore>())
      : ImplToExpandedFst<Impl>(
            std::make_shared<Impl>(
                fst,
                std::make_shared<Compactor>(
                    std::make_shared<ArcCompactor>(compactor), data),
                opts)) {}

  // If data is not nullptr, it is assumed to be already initialized.
  CompactFst(
      const Fst<Arc> &fst,
      std::shared_ptr<ArcCompactor> compactor,
      const CompactFstOptions &opts = CompactFstOptions(),
      std::shared_ptr<CompactStore> data = std::shared_ptr<CompactStore>())
      : ImplToExpandedFst<Impl>(
            std::make_shared<Impl>(fst,
                                   std::make_shared<Compactor>(compactor, data),
                                   opts)) {}

  // The following 2 constructors take as input two iterators delimiting a set
  // of (already) compacted transitions, starting with the transitions out of
  // the initial state. The format of the input differs for fixed out-degree
  // and variable out-degree compactors.
  //
  // - For fixed out-degree compactors, the final weight (encoded as a
  // compacted transition) needs to be given only for final states. All strings
  // (compactor of size 1) will be assume to be terminated by a final state
  // even when the final state is not implicitely given.
  //
  // - For variable out-degree compactors, the final weight (encoded as a
  // compacted transition) needs to be given for all states and must appeared
  // first in the list (for state s, final weight of s, followed by outgoing
  // transitons in s).
  //
  // These 2 constructors allows the direct construction of a CompactFst
  // without first creating a more memory-hungry regular FST. This is useful
  // when memory usage is severely constrained.
  template <class Iterator>
  explicit CompactFst(const Iterator &begin, const Iterator &end,
                      const ArcCompactor &compactor = ArcCompactor(),
                      const CompactFstOptions &opts = CompactFstOptions())
      : ImplToExpandedFst<Impl>(
            std::make_shared<Impl>(
                std::make_shared<Compactor>(
                    begin, end, std::make_shared<ArcCompactor>(compactor)),
                opts)) {}

  template <class Iterator>
  CompactFst(const Iterator &begin, const Iterator &end,
             std::shared_ptr<ArcCompactor> compactor,
             const CompactFstOptions &opts = CompactFstOptions())
      : ImplToExpandedFst<Impl>(
            std::make_shared<Impl>(
                std::make_shared<Compactor>(begin, end, compactor), opts)) {}

  // See Fst<>::Copy() for doc.
  CompactFst(
      const CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>
      &fst,
      bool safe = false)
      : ImplToExpandedFst<Impl>(fst, safe) {}

  // Get a copy of this CompactFst. See Fst<>::Copy() for further doc.
  CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore> *Copy(
      bool safe = false) const override {
    return new CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>(
        *this, safe);
  }

  // Read a CompactFst from an input stream; return nullptr on error
  static CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore> *Read(
      std::istream &strm, const FstReadOptions &opts) {
    auto *impl = Impl::Read(strm, opts);
    return impl ? new CompactFst<A, ArcCompactor, Unsigned, CompactStore,
                                 CacheStore>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  // Read a CompactFst from a file; return nullptr on error
  // Empty filename reads from standard input
  static CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore> *Read(
      const string &filename) {
    auto *impl = ImplToExpandedFst<Impl>::Read(filename);
    return impl ? new CompactFst<A, ArcCompactor, Unsigned, CompactStore,
                                 CacheStore>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return GetImpl()->Write(strm, opts);
  }

  bool Write(const string &filename) const override {
    return Fst<Arc>::WriteFile(filename);
  }

  template <class FST>
  static bool WriteFst(const FST &fst, const ArcCompactor &compactor,
                       std::ostream &strm, const FstWriteOptions &opts);

  void InitStateIterator(StateIteratorData<Arc> *data) const override {
    GetImpl()->InitStateIterator(data);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  MatcherBase<Arc> *InitMatcher(MatchType match_type) const override {
    return new SortedMatcher<
        CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>>(
        *this, match_type);
  }

  template <class Iterator>
  void SetCompactElements(const Iterator &b, const Iterator &e) {
    GetMutableImpl()->SetCompactor(std::make_shared<Compactor>(
        b, e, std::make_shared<ArcCompactor>()));
  }

 private:
  using ImplToFst<Impl, ExpandedFst<Arc>>::GetImpl;
  using ImplToFst<Impl, ExpandedFst<Arc>>::GetMutableImpl;

  explicit CompactFst(std::shared_ptr<Impl> impl)
      : ImplToExpandedFst<Impl>(impl) {}

  // Use overloading to extract the type of the argument.
  static Impl *GetImplIfCompactFst(
      const CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>
          &compact_fst) {
    return compact_fst.GetImpl();
  }

  // This does not give privileged treatment to subclasses of CompactFst.
  template <typename NonCompactFst>
  static Impl *GetImplIfCompactFst(const NonCompactFst &fst) {
    return nullptr;
  }

  CompactFst &operator=(const CompactFst &fst) = delete;
};

// Writes FST in Compact format, with a possible pass over the machine before
// writing to compute the number of states and arcs.
template <class A, class ArcCompactor, class Unsigned, class CompactStore,
          class CacheStore>
template <class FST>
bool CompactFst<A, ArcCompactor, Unsigned, CompactStore, CacheStore>::WriteFst(
    const FST &fst, const ArcCompactor &compactor, std::ostream &strm,
    const FstWriteOptions &opts) {
  using Arc = A;
  using Weight = typename A::Weight;
  using Element = typename ArcCompactor::Element;
  const auto file_version =
      opts.align ? Impl::kAlignedFileVersion : Impl::kFileVersion;
  size_t num_arcs = -1;
  size_t num_states = -1;
  auto first_pass_compactor = compactor;
  if (auto *impl = GetImplIfCompactFst(fst)) {
    num_arcs = impl->GetCompactor()->GetCompactStore()->NumArcs();
    num_states = impl->GetCompactor()->GetCompactStore()->NumStates();
    first_pass_compactor = *impl->GetCompactor()->GetArcCompactor();
  } else {
    // A first pass is needed to compute the state of the compactor, which
    // is saved ahead of the rest of the data structures. This unfortunately
    // means forcing a complete double compaction when writing in this format.
    // TODO(allauzen): eliminate mutable state from compactors.
    num_arcs = 0;
    num_states = 0;
    for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      ++num_states;
      if (fst.Final(s) != Weight::Zero()) {
        first_pass_compactor.Compact(
            s, Arc(kNoLabel, kNoLabel, fst.Final(s), kNoStateId));
      }
      for (ArcIterator<FST> aiter(fst, s); !aiter.Done(); aiter.Next()) {
        ++num_arcs;
        first_pass_compactor.Compact(s, aiter.Value());
      }
    }
  }
  FstHeader hdr;
  hdr.SetStart(fst.Start());
  hdr.SetNumStates(num_states);
  hdr.SetNumArcs(num_arcs);
  string type = "compact";
  if (sizeof(Unsigned) != sizeof(uint32)) {
    type += std::to_string(CHAR_BIT * sizeof(Unsigned));
  }
  type += "_";
  type += ArcCompactor::Type();
  if (CompactStore::Type() != "compact") {
    type += "_";
    type += CompactStore::Type();
  }
  const auto copy_properties = fst.Properties(kCopyProperties, true);
  if ((copy_properties & kError) || !compactor.Compatible(fst)) {
    FSTERROR() << "Fst incompatible with compactor";
    return false;
  }
  uint64 properties = copy_properties | Impl::kStaticProperties;
  internal::FstImpl<Arc>::WriteFstHeader(fst, strm, opts, file_version, type,
                                         properties, &hdr);
  first_pass_compactor.Write(strm);
  if (first_pass_compactor.Size() == -1) {
    if (opts.align && !AlignOutput(strm)) {
      LOG(ERROR) << "CompactFst::Write: Alignment failed: " << opts.source;
      return false;
    }
    Unsigned compacts = 0;
    for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      strm.write(reinterpret_cast<const char *>(&compacts), sizeof(compacts));
      if (fst.Final(s) != Weight::Zero()) {
        ++compacts;
      }
      compacts += fst.NumArcs(s);
    }
    strm.write(reinterpret_cast<const char *>(&compacts), sizeof(compacts));
  }
  if (opts.align && !AlignOutput(strm)) {
    LOG(ERROR) << "Could not align file during write after writing states";
  }
  const auto &second_pass_compactor = compactor;
  Element element;
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    if (fst.Final(s) != Weight::Zero()) {
      element = second_pass_compactor.Compact(
          s, A(kNoLabel, kNoLabel, fst.Final(s), kNoStateId));
      strm.write(reinterpret_cast<const char *>(&element), sizeof(element));
    }
    for (ArcIterator<FST> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      element = second_pass_compactor.Compact(s, aiter.Value());
      strm.write(reinterpret_cast<const char *>(&element), sizeof(element));
    }
  }
  strm.flush();
  if (!strm) {
    LOG(ERROR) << "CompactFst write failed: " << opts.source;
    return false;
  }
  return true;
}

// Specialization for CompactFst; see generic version in fst.h for sample
// usage (but use the CompactFst type!). This version should inline.
template <class Arc, class ArcCompactor, class Unsigned, class CompactStore,
          class CacheStore>
class StateIterator<
    CompactFst<Arc, ArcCompactor, Unsigned, CompactStore, CacheStore>> {
 public:
  using StateId = typename Arc::StateId;

  explicit StateIterator(
      const CompactFst<Arc, ArcCompactor, Unsigned, CompactStore,
                       CacheStore> &fst)
      : nstates_(fst.GetImpl()->NumStates()), s_(0) {}

  bool Done() const { return s_ >= nstates_; }

  StateId Value() const { return s_; }

  void Next() { ++s_; }

  void Reset() { s_ = 0; }

 private:
  StateId nstates_;
  StateId s_;
};

// Specialization for CompactFst. Never caches,
// always iterates over the underlying compact elements.
template <class Arc, class ArcCompactor, class Unsigned,
          class CompactStore, class CacheStore>
class ArcIterator<CompactFst<
    Arc, ArcCompactor, Unsigned, CompactStore, CacheStore>> {
 public:
  using StateId = typename Arc::StateId;
  using Element = typename ArcCompactor::Element;
  using Compactor = DefaultCompactor<ArcCompactor, Unsigned, CompactStore>;
  using State = typename Compactor::State;

  ArcIterator(const CompactFst<Arc, ArcCompactor, Unsigned, CompactStore,
                               CacheStore> &fst,
              StateId s)
      : state_(fst.GetImpl()->GetCompactor(), s),
        pos_(0),
        flags_(kArcValueFlags) {}

  bool Done() const { return pos_ >= state_.NumArcs(); }

  const Arc &Value() const {
    arc_ = state_.GetArc(pos_, flags_);
    return arc_;
  }

  void Next() { ++pos_; }

  size_t Position() const { return pos_; }

  void Reset() { pos_ = 0; }

  void Seek(size_t pos) { pos_ = pos; }

  uint32 Flags() const { return flags_; }

  void SetFlags(uint32 f, uint32 m) {
    flags_ &= ~m;
    flags_ |= (f & kArcValueFlags);
  }

 private:
  State state_;
  size_t pos_;
  mutable Arc arc_;
  uint32 flags_;
};

// ArcCompactor for unweighted string FSTs.
template <class A>
class StringCompactor {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Element = Label;

  Element Compact(StateId s, const Arc &arc) const { return arc.ilabel; }

  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    return Arc(p, p, Weight::One(), p != kNoLabel ? s + 1 : kNoStateId);
  }

  constexpr ssize_t Size() const { return 1; }

  constexpr uint64 Properties() const {
    return kString | kAcceptor | kUnweighted;
  }

  bool Compatible(const Fst<Arc> &fst) const {
    const auto props = Properties();
    return fst.Properties(props, true) == props;
  }

  static const string &Type() {
    static const string *const type = new string("string");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static StringCompactor *Read(std::istream &strm) {
    return new StringCompactor;
  }
};

// ArcCompactor for weighted string FSTs.
template <class A>
class WeightedStringCompactor {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Element = std::pair<Label, Weight>;

  Element Compact(StateId s, const Arc &arc) const {
    return std::make_pair(arc.ilabel, arc.weight);
  }

  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    return Arc(p.first, p.first, p.second,
               p.first != kNoLabel ? s + 1 : kNoStateId);
  }

  constexpr ssize_t Size() const { return 1; }

  constexpr uint64 Properties() const { return kString | kAcceptor; }

  bool Compatible(const Fst<Arc> &fst) const {
    const auto props = Properties();
    return fst.Properties(props, true) == props;
  }

  static const string &Type() {
    static const string *const type = new string("weighted_string");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static WeightedStringCompactor *Read(std::istream &strm) {
    return new WeightedStringCompactor;
  }
};

// ArcCompactor for unweighted acceptor FSTs.
template <class A>
class UnweightedAcceptorCompactor {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Element = std::pair<Label, StateId>;

  Element Compact(StateId s, const Arc &arc) const {
    return std::make_pair(arc.ilabel, arc.nextstate);
  }

  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    return Arc(p.first, p.first, Weight::One(), p.second);
  }

  constexpr ssize_t Size() const { return -1; }

  constexpr uint64 Properties() const { return kAcceptor | kUnweighted; }

  bool Compatible(const Fst<Arc> &fst) const {
    const auto props = Properties();
    return fst.Properties(props, true) == props;
  }

  static const string &Type() {
    static const string *const type = new string("unweighted_acceptor");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static UnweightedAcceptorCompactor *Read(std::istream &istrm) {
    return new UnweightedAcceptorCompactor;
  }
};

// ArcCompactor for weighted acceptor FSTs.
template <class A>
class AcceptorCompactor {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Element = std::pair<std::pair<Label, Weight>, StateId>;

  Element Compact(StateId s, const Arc &arc) const {
    return std::make_pair(std::make_pair(arc.ilabel, arc.weight),
                          arc.nextstate);
  }

  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    return Arc(p.first.first, p.first.first, p.first.second, p.second);
  }

  constexpr ssize_t Size() const { return -1; }

  constexpr uint64 Properties() const { return kAcceptor; }

  bool Compatible(const Fst<Arc> &fst) const {
    const auto props = Properties();
    return fst.Properties(props, true) == props;
  }

  static const string &Type() {
    static const string *const type = new string("acceptor");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static AcceptorCompactor *Read(std::istream &strm) {
    return new AcceptorCompactor;
  }
};

// ArcCompactor for unweighted FSTs.
template <class A>
class UnweightedCompactor {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Element = std::pair<std::pair<Label, Label>, StateId>;

  Element Compact(StateId s, const Arc &arc) const {
    return std::make_pair(std::make_pair(arc.ilabel, arc.olabel),
                          arc.nextstate);
  }

  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    return Arc(p.first.first, p.first.second, Weight::One(), p.second);
  }

  constexpr ssize_t Size() const { return -1; }

  constexpr uint64 Properties() const { return kUnweighted; }

  bool Compatible(const Fst<Arc> &fst) const {
    const auto props = Properties();
    return fst.Properties(props, true) == props;
  }

  static const string &Type() {
    static const string *const type = new string("unweighted");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static UnweightedCompactor *Read(std::istream &strm) {
    return new UnweightedCompactor;
  }
};

template <class Arc, class Unsigned /* = uint32 */>
using CompactStringFst = CompactFst<Arc, StringCompactor<Arc>, Unsigned>;

template <class Arc, class Unsigned /* = uint32 */>
using CompactWeightedStringFst =
    CompactFst<Arc, WeightedStringCompactor<Arc>, Unsigned>;

template <class Arc, class Unsigned /* = uint32 */>
using CompactAcceptorFst = CompactFst<Arc, AcceptorCompactor<Arc>, Unsigned>;

template <class Arc, class Unsigned /* = uint32 */>
using CompactUnweightedFst =
    CompactFst<Arc, UnweightedCompactor<Arc>, Unsigned>;

template <class Arc, class Unsigned /* = uint32 */>
using CompactUnweightedAcceptorFst =
    CompactFst<Arc, UnweightedAcceptorCompactor<Arc>, Unsigned>;

using StdCompactStringFst = CompactStringFst<StdArc, uint32>;

using StdCompactWeightedStringFst = CompactWeightedStringFst<StdArc, uint32>;

using StdCompactAcceptorFst = CompactAcceptorFst<StdArc, uint32>;

using StdCompactUnweightedFst = CompactUnweightedFst<StdArc, uint32>;

using StdCompactUnweightedAcceptorFst =
    CompactUnweightedAcceptorFst<StdArc, uint32>;

}  // namespace fst

#endif  // FST_COMPACT_FST_H_
