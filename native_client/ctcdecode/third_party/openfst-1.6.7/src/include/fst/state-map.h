// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to map over/transform states e.g., sort transitions.
//
// Consider using when operation does not change the number of states.

#ifndef FST_STATE_MAP_H_
#define FST_STATE_MAP_H_

#include <algorithm>
#include <string>
#include <utility>

#include <fst/log.h>

#include <fst/arc-map.h>
#include <fst/cache.h>
#include <fst/mutable-fst.h>


namespace fst {

// StateMapper Interface. The class determines how states are mapped; useful for
// implementing operations that do not change the number of states.
//
// class StateMapper {
//  public:
//   using FromArc = A;
//   using ToArc = B;
//
//   // Typical constructor.
//   StateMapper(const Fst<A> &fst);
//
//   // Required copy constructor that allows updating FST argument;
//   // pass only if relevant and changed.
//   StateMapper(const StateMapper &mapper, const Fst<A> *fst = 0);
//
//   // Specifies initial state of result.
//   B::StateId Start() const;
//   // Specifies state's final weight in result.
//   B::Weight Final(B::StateId state) const;
//
//   // These methods iterate through a state's arcs in result.
//
//   // Specifies state to iterate over.
//   void SetState(B::StateId state);
//
//   // End of arcs?
//   bool Done() const;
//
//   // Current arc.
//   const B &Value() const;
//
//   // Advances to next arc (when !Done)
//   void Next();
//
//   // Specifies input symbol table action the mapper requires (see above).
//   MapSymbolsAction InputSymbolsAction() const;
//
//   // Specifies output symbol table action the mapper requires (see above).
//   MapSymbolsAction OutputSymbolsAction() const;
//
//   // This specifies the known properties of an FST mapped by this
//   // mapper. It takes as argument the input FST's known properties.
//   uint64 Properties(uint64 props) const;
// };
//
// We include a various state map versions below. One dimension of variation is
// whether the mapping mutates its input, writes to a new result FST, or is an
// on-the-fly Fst. Another dimension is how we pass the mapper. We allow passing
// the mapper by pointer for cases that we need to change the state of the
// user's mapper. We also include map versions that pass the mapper by value or
// const reference when this suffices.

// Maps an arc type A using a mapper function object C, passed by pointer. This
// version modifies the input FST.
template <class A, class C>
void StateMap(MutableFst<A> *fst, C *mapper) {
  if (mapper->InputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
    fst->SetInputSymbols(nullptr);
  }
  if (mapper->OutputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
    fst->SetOutputSymbols(nullptr);
  }
  if (fst->Start() == kNoStateId) return;
  const auto props = fst->Properties(kFstProperties, false);
  fst->SetStart(mapper->Start());
  for (StateIterator<Fst<A>> siter(*fst); !siter.Done(); siter.Next()) {
    const auto state = siter.Value();
    mapper->SetState(state);
    fst->DeleteArcs(state);
    for (; !mapper->Done(); mapper->Next()) {
      fst->AddArc(state, mapper->Value());
    }
    fst->SetFinal(state, mapper->Final(state));
  }
  fst->SetProperties(mapper->Properties(props), kFstProperties);
}

// Maps an arc type A using a mapper function object C, passed by value.
// This version modifies the input FST.
template <class A, class C>
void StateMap(MutableFst<A> *fst, C mapper) {
  StateMap(fst, &mapper);
}

// Maps an arc type A to an arc type B using mapper functor C, passed by
// pointer. This version writes to an output FST.
template <class A, class B, class C>
void StateMap(const Fst<A> &ifst, MutableFst<B> *ofst, C *mapper) {
  ofst->DeleteStates();
  if (mapper->InputSymbolsAction() == MAP_COPY_SYMBOLS) {
    ofst->SetInputSymbols(ifst.InputSymbols());
  } else if (mapper->InputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
    ofst->SetInputSymbols(nullptr);
  }
  if (mapper->OutputSymbolsAction() == MAP_COPY_SYMBOLS) {
    ofst->SetOutputSymbols(ifst.OutputSymbols());
  } else if (mapper->OutputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
    ofst->SetOutputSymbols(nullptr);
  }
  const auto iprops = ifst.Properties(kCopyProperties, false);
  if (ifst.Start() == kNoStateId) {
    if (iprops & kError) ofst->SetProperties(kError, kError);
    return;
  }
  // Adds all states.
  if (ifst.Properties(kExpanded, false)) ofst->ReserveStates(CountStates(ifst));
  for (StateIterator<Fst<A>> siter(ifst); !siter.Done(); siter.Next()) {
    ofst->AddState();
  }
  ofst->SetStart(mapper->Start());
  for (StateIterator<Fst<A>> siter(ifst); !siter.Done(); siter.Next()) {
    const auto state = siter.Value();
    mapper->SetState(state);
    for (; !mapper->Done(); mapper->Next()) {
      ofst->AddArc(state, mapper->Value());
    }
    ofst->SetFinal(state, mapper->Final(state));
  }
  const auto oprops = ofst->Properties(kFstProperties, false);
  ofst->SetProperties(mapper->Properties(iprops) | oprops, kFstProperties);
}

// Maps an arc type A to an arc type B using mapper functor object C, passed by
// value. This version writes to an output FST.
template <class A, class B, class C>
void StateMap(const Fst<A> &ifst, MutableFst<B> *ofst, C mapper) {
  StateMap(ifst, ofst, &mapper);
}

using StateMapFstOptions = CacheOptions;

template <class A, class B, class C>
class StateMapFst;

// Facade around StateIteratorBase<A> inheriting from StateIteratorBase<B>.
template <class A, class B>
class StateMapStateIteratorBase : public StateIteratorBase<B> {
 public:
  using Arc = B;
  using StateId = typename Arc::StateId;

  explicit StateMapStateIteratorBase(StateIteratorBase<A> *base)
      : base_(base) {}

  bool Done() const final { return base_->Done(); }

  StateId Value() const final { return base_->Value(); }

  void Next() final { base_->Next(); }

  void Reset() final { base_->Reset(); }

 private:
  std::unique_ptr<StateIteratorBase<A>> base_;

  StateMapStateIteratorBase() = delete;
};

namespace internal {

// Implementation of delayed StateMapFst.
template <class A, class B, class C>
class StateMapFstImpl : public CacheImpl<B> {
 public:
  using Arc = B;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<B>::SetType;
  using FstImpl<B>::SetProperties;
  using FstImpl<B>::SetInputSymbols;
  using FstImpl<B>::SetOutputSymbols;

  using CacheImpl<B>::PushArc;
  using CacheImpl<B>::HasArcs;
  using CacheImpl<B>::HasFinal;
  using CacheImpl<B>::HasStart;
  using CacheImpl<B>::SetArcs;
  using CacheImpl<B>::SetFinal;
  using CacheImpl<B>::SetStart;

  friend class StateIterator<StateMapFst<A, B, C>>;

  StateMapFstImpl(const Fst<A> &fst, const C &mapper,
                  const StateMapFstOptions &opts)
      : CacheImpl<B>(opts),
        fst_(fst.Copy()),
        mapper_(new C(mapper, fst_.get())),
        own_mapper_(true) {
    Init();
  }

  StateMapFstImpl(const Fst<A> &fst, C *mapper, const StateMapFstOptions &opts)
      : CacheImpl<B>(opts),
        fst_(fst.Copy()),
        mapper_(mapper),
        own_mapper_(false) {
    Init();
  }

  StateMapFstImpl(const StateMapFstImpl<A, B, C> &impl)
      : CacheImpl<B>(impl),
        fst_(impl.fst_->Copy(true)),
        mapper_(new C(*impl.mapper_, fst_.get())),
        own_mapper_(true) {
    Init();
  }

  ~StateMapFstImpl() override {
    if (own_mapper_) delete mapper_;
  }

  StateId Start() {
    if (!HasStart()) SetStart(mapper_->Start());
    return CacheImpl<B>::Start();
  }

  Weight Final(StateId state) {
    if (!HasFinal(state)) SetFinal(state, mapper_->Final(state));
    return CacheImpl<B>::Final(state);
  }

  size_t NumArcs(StateId state) {
    if (!HasArcs(state)) Expand(state);
    return CacheImpl<B>::NumArcs(state);
  }

  size_t NumInputEpsilons(StateId state) {
    if (!HasArcs(state)) Expand(state);
    return CacheImpl<B>::NumInputEpsilons(state);
  }

  size_t NumOutputEpsilons(StateId state) {
    if (!HasArcs(state)) Expand(state);
    return CacheImpl<B>::NumOutputEpsilons(state);
  }

  void InitStateIterator(StateIteratorData<B> *datb) const {
    StateIteratorData<A> data;
    fst_->InitStateIterator(&data);
    datb->base = data.base ? new StateMapStateIteratorBase<A, B>(data.base)
        : nullptr;
    datb->nstates = data.nstates;
  }

  void InitArcIterator(StateId state, ArcIteratorData<B> *data) {
    if (!HasArcs(state)) Expand(state);
    CacheImpl<B>::InitArcIterator(state, data);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) && (fst_->Properties(kError, false) ||
                            (mapper_->Properties(0) & kError))) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  void Expand(StateId state) {
    // Adds exiting arcs.
    for (mapper_->SetState(state); !mapper_->Done(); mapper_->Next()) {
      PushArc(state, mapper_->Value());
    }
    SetArcs(state);
  }

  const Fst<A> *GetFst() const { return fst_.get(); }

 private:
  void Init() {
    SetType("statemap");
    if (mapper_->InputSymbolsAction() == MAP_COPY_SYMBOLS) {
      SetInputSymbols(fst_->InputSymbols());
    } else if (mapper_->InputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
      SetInputSymbols(nullptr);
    }
    if (mapper_->OutputSymbolsAction() == MAP_COPY_SYMBOLS) {
      SetOutputSymbols(fst_->OutputSymbols());
    } else if (mapper_->OutputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
      SetOutputSymbols(nullptr);
    }
    const auto props = fst_->Properties(kCopyProperties, false);
    SetProperties(mapper_->Properties(props));
  }

  std::unique_ptr<const Fst<A>> fst_;
  C *mapper_;
  bool own_mapper_;
};

}  // namespace internal

// Maps an arc type A to an arc type B using Mapper function object
// C. This version is a delayed FST.
template <class A, class B, class C>
class StateMapFst : public ImplToFst<internal::StateMapFstImpl<A, B, C>> {
 public:
  friend class ArcIterator<StateMapFst<A, B, C>>;

  using Arc = B;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using Store = DefaultCacheStore<Arc>;
  using State = typename Store::State;
  using Impl = internal::StateMapFstImpl<A, B, C>;

  StateMapFst(const Fst<A> &fst, const C &mapper,
              const StateMapFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, mapper, opts)) {}

  StateMapFst(const Fst<A> &fst, C *mapper, const StateMapFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, mapper, opts)) {}

  StateMapFst(const Fst<A> &fst, const C &mapper)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, mapper, StateMapFstOptions())) {}

  StateMapFst(const Fst<A> &fst, C *mapper)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, mapper, StateMapFstOptions())) {}

  // See Fst<>::Copy() for doc.
  StateMapFst(const StateMapFst<A, B, C> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this StateMapFst. See Fst<>::Copy() for further doc.
  StateMapFst<A, B, C> *Copy(bool safe = false) const override {
    return new StateMapFst<A, B, C>(*this, safe);
  }

  void InitStateIterator(StateIteratorData<B> *data) const override {
    GetImpl()->InitStateIterator(data);
  }

  void InitArcIterator(StateId state, ArcIteratorData<B> *data) const override {
    GetMutableImpl()->InitArcIterator(state, data);
  }

 protected:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

 private:
  StateMapFst &operator=(const StateMapFst &) = delete;
};

// Specialization for StateMapFst.
template <class A, class B, class C>
class ArcIterator<StateMapFst<A, B, C>>
    : public CacheArcIterator<StateMapFst<A, B, C>> {
 public:
  using StateId = typename A::StateId;

  ArcIterator(const StateMapFst<A, B, C> &fst, StateId state)
      : CacheArcIterator<StateMapFst<A, B, C>>(fst.GetMutableImpl(), state) {
    if (!fst.GetImpl()->HasArcs(state)) fst.GetMutableImpl()->Expand(state);
  }
};

// Utility mappers.

// Mapper that returns its input.
template <class Arc>
class IdentityStateMapper {
 public:
  using FromArc = Arc;
  using ToArc = Arc;

  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit IdentityStateMapper(const Fst<Arc> &fst) : fst_(fst) {}

  // Allows updating FST argument; pass only if changed.
  IdentityStateMapper(const IdentityStateMapper<Arc> &mapper,
                      const Fst<Arc> *fst = nullptr)
      : fst_(fst ? *fst : mapper.fst_) {}

  StateId Start() const { return fst_.Start(); }

  Weight Final(StateId state) const { return fst_.Final(state); }

  void SetState(StateId state) {
    aiter_.reset(new ArcIterator<Fst<Arc>>(fst_, state));
  }

  bool Done() const { return aiter_->Done(); }

  const Arc &Value() const { return aiter_->Value(); }

  void Next() { aiter_->Next(); }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }

 private:
  const Fst<Arc> &fst_;
  std::unique_ptr<ArcIterator<Fst<Arc>>> aiter_;
};

template <class Arc>
class ArcSumMapper {
 public:
  using FromArc = Arc;
  using ToArc = Arc;

  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit ArcSumMapper(const Fst<Arc> &fst) : fst_(fst), i_(0) {}

  // Allows updating FST argument; pass only if changed.
  ArcSumMapper(const ArcSumMapper<Arc> &mapper, const Fst<Arc> *fst = nullptr)
      : fst_(fst ? *fst : mapper.fst_), i_(0) {}

  StateId Start() const { return fst_.Start(); }

  Weight Final(StateId state) const { return fst_.Final(state); }

  void SetState(StateId state) {
    i_ = 0;
    arcs_.clear();
    arcs_.reserve(fst_.NumArcs(state));
    for (ArcIterator<Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      arcs_.push_back(aiter.Value());
    }
    // First sorts the exiting arcs by input label, output label and destination
    // state and then sums weights of arcs with the same input label, output
    // label, and destination state.
    std::sort(arcs_.begin(), arcs_.end(), comp_);
    size_t narcs = 0;
    for (const auto &arc : arcs_) {
      if (narcs > 0 && equal_(arc, arcs_[narcs - 1])) {
        arcs_[narcs - 1].weight = Plus(arcs_[narcs - 1].weight, arc.weight);
      } else {
        arcs_[narcs] = arc;
        ++narcs;
      }
    }
    arcs_.resize(narcs);
  }

  bool Done() const { return i_ >= arcs_.size(); }

  const Arc &Value() const { return arcs_[i_]; }

  void Next() { ++i_; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kArcSortProperties & kDeleteArcsProperties &
           kWeightInvariantProperties;
  }

 private:
  struct Compare {
    bool operator()(const Arc &x, const Arc &y) const {
      if (x.ilabel < y.ilabel) return true;
      if (x.ilabel > y.ilabel) return false;
      if (x.olabel < y.olabel) return true;
      if (x.olabel > y.olabel) return false;
      if (x.nextstate < y.nextstate) return true;
      if (x.nextstate > y.nextstate) return false;
      return false;
    }
  };

  struct Equal {
    bool operator()(const Arc &x, const Arc &y) const {
      return (x.ilabel == y.ilabel && x.olabel == y.olabel &&
              x.nextstate == y.nextstate);
    }
  };

  const Fst<Arc> &fst_;
  Compare comp_;
  Equal equal_;
  std::vector<Arc> arcs_;
  ssize_t i_;  // Current arc position.

  ArcSumMapper &operator=(const ArcSumMapper &) = delete;
};

template <class Arc>
class ArcUniqueMapper {
 public:
  using FromArc = Arc;
  using ToArc = Arc;

  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit ArcUniqueMapper(const Fst<Arc> &fst) : fst_(fst), i_(0) {}

  // Allows updating FST argument; pass only if changed.
  ArcUniqueMapper(const ArcUniqueMapper<Arc> &mapper,
                  const Fst<Arc> *fst = nullptr)
      : fst_(fst ? *fst : mapper.fst_), i_(0) {}

  StateId Start() const { return fst_.Start(); }

  Weight Final(StateId state) const { return fst_.Final(state); }

  void SetState(StateId state) {
    i_ = 0;
    arcs_.clear();
    arcs_.reserve(fst_.NumArcs(state));
    for (ArcIterator<Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      arcs_.push_back(aiter.Value());
    }
    // First sorts the exiting arcs by input label, output label and destination
    // state and then uniques identical arcs.
    std::sort(arcs_.begin(), arcs_.end(), comp_);
    arcs_.erase(std::unique(arcs_.begin(), arcs_.end(), equal_), arcs_.end());
  }

  bool Done() const { return i_ >= arcs_.size(); }

  const Arc &Value() const { return arcs_[i_]; }

  void Next() { ++i_; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kArcSortProperties & kDeleteArcsProperties;
  }

 private:
  struct Compare {
    bool operator()(const Arc &x, const Arc &y) const {
      if (x.ilabel < y.ilabel) return true;
      if (x.ilabel > y.ilabel) return false;
      if (x.olabel < y.olabel) return true;
      if (x.olabel > y.olabel) return false;
      if (x.nextstate < y.nextstate) return true;
      if (x.nextstate > y.nextstate) return false;
      return false;
    }
  };

  struct Equal {
    bool operator()(const Arc &x, const Arc &y) const {
      return (x.ilabel == y.ilabel && x.olabel == y.olabel &&
              x.nextstate == y.nextstate && x.weight == y.weight);
    }
  };

  const Fst<Arc> &fst_;
  Compare comp_;
  Equal equal_;
  std::vector<Arc> arcs_;
  size_t i_;  // Current arc position.

  ArcUniqueMapper &operator=(const ArcUniqueMapper &) = delete;
};

// Useful aliases when using StdArc.

using StdArcSumMapper = ArcSumMapper<StdArc>;

using StdArcUniqueMapper = ArcUniqueMapper<StdArc>;

}  // namespace fst

#endif  // FST_STATE_MAP_H_
