// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to map over/transform arcs e.g., change semirings or
// implement project/invert. Consider using when operation does
// not change the number of arcs (except possibly superfinal arcs).

#ifndef FST_ARC_MAP_H_
#define FST_ARC_MAP_H_

#include <string>
#include <unordered_map>
#include <utility>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/mutable-fst.h>


namespace fst {

// Determines how final weights are mapped.
enum MapFinalAction {
  // A final weight is mapped into a final weight. An error is raised if this
  // is not possible.
  MAP_NO_SUPERFINAL,
  // A final weight is mapped to an arc to the superfinal state when the result
  // cannot be represented as a final weight. The superfinal state will be
  // added only if it is needed.
  MAP_ALLOW_SUPERFINAL,
  // A final weight is mapped to an arc to the superfinal state unless the
  // result can be represented as a final weight of weight Zero(). The
  // superfinal state is always added (if the input is not the empty FST).
  MAP_REQUIRE_SUPERFINAL
};

// Determines how symbol tables are mapped.
enum MapSymbolsAction {
  // Symbols should be cleared in the result by the map.
  MAP_CLEAR_SYMBOLS,
  // Symbols should be copied from the input FST by the map.
  MAP_COPY_SYMBOLS,
  // Symbols should not be modified in the result by the map itself.
  // (They may set by the mapper).
  MAP_NOOP_SYMBOLS
};

// The ArcMapper interfaces defines how arcs and final weights are mapped.
// This is useful for implementing operations that do not change the number of
// arcs (expect possibly superfinal arcs).
//
// template <class A, class B>
// class ArcMapper {
//  public:
//   using FromArc = A;
//   using ToArc = B;
//
//   // Maps an arc type FromArc to arc type ToArc.
//   ToArc operator()(const FromArc &arc);
//
//   // Specifies final action the mapper requires (see above).
//   // The mapper will be passed final weights as arcs of the form
//   // Arc(0, 0, weight, kNoStateId).
//   MapFinalAction FinalAction() const;
//
//   // Specifies input symbol table action the mapper requires (see above).
//   MapSymbolsAction InputSymbolsAction() const;
//
//   // Specifies output symbol table action the mapper requires (see above).
//   MapSymbolsAction OutputSymbolsAction() const;
//
//   // This specifies the known properties of an FST mapped by this mapper. It
//   takes as argument the input FSTs's known properties.
//   uint64 Properties(uint64 props) const;
// };
//
// The ArcMap functions and classes below will use the FinalAction()
// method of the mapper to determine how to treat final weights, e.g., whether
// to add a superfinal state. They will use the Properties() method to set the
// result FST properties.
//
// We include a various map versions below. One dimension of variation is
// whether the mapping mutates its input, writes to a new result FST, or is an
// on-the-fly FST. Another dimension is how we pass the mapper. We allow passing
// the mapper by pointer for cases that we need to change the state of the
// user's mapper.  This is the case with the EncodeMapper, which is reused
// during decoding. We also include map versions that pass the mapper by value
// or const reference when this suffices.

// Maps an arc type A using a mapper function object C, passed
// by pointer.  This version modifies its Fst input.
template <class A, class C>
void ArcMap(MutableFst<A> *fst, C *mapper) {
  using FromArc = A;
  using ToArc = A;
  using StateId = typename FromArc::StateId;
  using Weight = typename FromArc::Weight;
  if (mapper->InputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
    fst->SetInputSymbols(nullptr);
  }
  if (mapper->OutputSymbolsAction() == MAP_CLEAR_SYMBOLS) {
    fst->SetOutputSymbols(nullptr);
  }
  if (fst->Start() == kNoStateId) return;
  const auto props = fst->Properties(kFstProperties, false);
  const auto final_action = mapper->FinalAction();
  auto superfinal = kNoStateId;
  if (final_action == MAP_REQUIRE_SUPERFINAL) {
    superfinal = fst->AddState();
    fst->SetFinal(superfinal, Weight::One());
  }
  for (StateIterator<MutableFst<FromArc>> siter(*fst); !siter.Done();
       siter.Next()) {
    const auto state = siter.Value();
    for (MutableArcIterator<MutableFst<FromArc>> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      aiter.SetValue((*mapper)(arc));
    }
    switch (final_action) {
      case MAP_NO_SUPERFINAL:
      default: {
        const FromArc arc(0, 0, fst->Final(state), kNoStateId);
        const auto final_arc = (*mapper)(arc);
        if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
          FSTERROR() << "ArcMap: Non-zero arc labels for superfinal arc";
          fst->SetProperties(kError, kError);
        }
        fst->SetFinal(state, final_arc.weight);
        break;
      }
      case MAP_ALLOW_SUPERFINAL: {
        if (state != superfinal) {
          const FromArc arc(0, 0, fst->Final(state), kNoStateId);
          auto final_arc = (*mapper)(arc);
          if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
            // Add a superfinal state if not already done.
            if (superfinal == kNoStateId) {
              superfinal = fst->AddState();
              fst->SetFinal(superfinal, Weight::One());
            }
            final_arc.nextstate = superfinal;
            fst->AddArc(state, final_arc);
            fst->SetFinal(state, Weight::Zero());
          } else {
            fst->SetFinal(state, final_arc.weight);
          }
        }
        break;
      }
      case MAP_REQUIRE_SUPERFINAL: {
        if (state != superfinal) {
          const FromArc arc(0, 0, fst->Final(state), kNoStateId);
          const auto final_arc = (*mapper)(arc);
          if (final_arc.ilabel != 0 || final_arc.olabel != 0 ||
              final_arc.weight != Weight::Zero()) {
            fst->AddArc(state, ToArc(final_arc.ilabel, final_arc.olabel,
                                     final_arc.weight, superfinal));
          }
          fst->SetFinal(state, Weight::Zero());
        }
        break;
      }
    }
  }
  fst->SetProperties(mapper->Properties(props), kFstProperties);
}

// Maps an arc type A using a mapper function object C, passed by value. This
// version modifies its FST input.
template <class A, class C>
void ArcMap(MutableFst<A> *fst, C mapper) {
  ArcMap(fst, &mapper);
}

// Maps an arc type A to an arc type B using mapper function object C,
// passed by pointer. This version writes the mapped input FST to an
// output MutableFst.
template <class A, class B, class C>
void ArcMap(const Fst<A> &ifst, MutableFst<B> *ofst, C *mapper) {
  using FromArc = A;
  using StateId = typename FromArc::StateId;
  using Weight = typename FromArc::Weight;
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
  const auto final_action = mapper->FinalAction();
  if (ifst.Properties(kExpanded, false)) {
    ofst->ReserveStates(
        CountStates(ifst) + final_action == MAP_NO_SUPERFINAL ? 0 : 1);
  }
  // Adds all states.
  for (StateIterator<Fst<A>> siter(ifst); !siter.Done(); siter.Next()) {
    ofst->AddState();
  }
  StateId superfinal = kNoStateId;
  if (final_action == MAP_REQUIRE_SUPERFINAL) {
    superfinal = ofst->AddState();
    ofst->SetFinal(superfinal, B::Weight::One());
  }
  for (StateIterator<Fst<A>> siter(ifst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    if (s == ifst.Start()) ofst->SetStart(s);
    ofst->ReserveArcs(s, ifst.NumArcs(s));
    for (ArcIterator<Fst<A>> aiter(ifst, s); !aiter.Done(); aiter.Next()) {
      ofst->AddArc(s, (*mapper)(aiter.Value()));
    }
    switch (final_action) {
      case MAP_NO_SUPERFINAL:
      default: {
        B final_arc = (*mapper)(A(0, 0, ifst.Final(s), kNoStateId));
        if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
          FSTERROR() << "ArcMap: Non-zero arc labels for superfinal arc";
          ofst->SetProperties(kError, kError);
        }
        ofst->SetFinal(s, final_arc.weight);
        break;
      }
      case MAP_ALLOW_SUPERFINAL: {
        B final_arc = (*mapper)(A(0, 0, ifst.Final(s), kNoStateId));
        if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
          // Add a superfinal state if not already done.
          if (superfinal == kNoStateId) {
            superfinal = ofst->AddState();
            ofst->SetFinal(superfinal, B::Weight::One());
          }
          final_arc.nextstate = superfinal;
          ofst->AddArc(s, final_arc);
          ofst->SetFinal(s, B::Weight::Zero());
        } else {
          ofst->SetFinal(s, final_arc.weight);
        }
        break;
      }
      case MAP_REQUIRE_SUPERFINAL: {
        B final_arc = (*mapper)(A(0, 0, ifst.Final(s), kNoStateId));
        if (final_arc.ilabel != 0 || final_arc.olabel != 0 ||
            final_arc.weight != B::Weight::Zero()) {
          ofst->AddArc(s, B(final_arc.ilabel, final_arc.olabel,
                            final_arc.weight, superfinal));
        }
        ofst->SetFinal(s, B::Weight::Zero());
        break;
      }
    }
  }
  const auto oprops = ofst->Properties(kFstProperties, false);
  ofst->SetProperties(mapper->Properties(iprops) | oprops, kFstProperties);
}

// Maps an arc type A to an arc type B using mapper function
// object C, passed by value. This version writes the mapped input
// Fst to an output MutableFst.
template <class A, class B, class C>
void ArcMap(const Fst<A> &ifst, MutableFst<B> *ofst, C mapper) {
  ArcMap(ifst, ofst, &mapper);
}

struct ArcMapFstOptions : public CacheOptions {
  // ArcMapFst default caching behaviour is to do no caching. Most mappers are
  // cheap and therefore we save memory by not doing caching.
  ArcMapFstOptions() : CacheOptions(true, 0) {}

  explicit ArcMapFstOptions(const CacheOptions &opts) : CacheOptions(opts) {}
};

template <class A, class B, class C>
class ArcMapFst;

namespace internal {

// Implementation of delayed ArcMapFst.
template <class A, class B, class C>
class ArcMapFstImpl : public CacheImpl<B> {
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

  friend class StateIterator<ArcMapFst<A, B, C>>;

  ArcMapFstImpl(const Fst<A> &fst, const C &mapper,
                const ArcMapFstOptions &opts)
      : CacheImpl<B>(opts),
        fst_(fst.Copy()),
        mapper_(new C(mapper)),
        own_mapper_(true),
        superfinal_(kNoStateId),
        nstates_(0) {
    Init();
  }

  ArcMapFstImpl(const Fst<A> &fst, C *mapper, const ArcMapFstOptions &opts)
      : CacheImpl<B>(opts),
        fst_(fst.Copy()),
        mapper_(mapper),
        own_mapper_(false),
        superfinal_(kNoStateId),
        nstates_(0) {
    Init();
  }

  ArcMapFstImpl(const ArcMapFstImpl<A, B, C> &impl)
      : CacheImpl<B>(impl),
        fst_(impl.fst_->Copy(true)),
        mapper_(new C(*impl.mapper_)),
        own_mapper_(true),
        superfinal_(kNoStateId),
        nstates_(0) {
    Init();
  }

  ~ArcMapFstImpl() override {
    if (own_mapper_) delete mapper_;
  }

  StateId Start() {
    if (!HasStart()) SetStart(FindOState(fst_->Start()));
    return CacheImpl<B>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      switch (final_action_) {
        case MAP_NO_SUPERFINAL:
        default: {
          const auto final_arc =
              (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
          if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
            FSTERROR() << "ArcMapFst: Non-zero arc labels for superfinal arc";
            SetProperties(kError, kError);
          }
          SetFinal(s, final_arc.weight);
          break;
        }
        case MAP_ALLOW_SUPERFINAL: {
          if (s == superfinal_) {
            SetFinal(s, Weight::One());
          } else {
            const auto final_arc =
                (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
            if (final_arc.ilabel == 0 && final_arc.olabel == 0) {
              SetFinal(s, final_arc.weight);
            } else {
              SetFinal(s, Weight::Zero());
            }
          }
          break;
        }
        case MAP_REQUIRE_SUPERFINAL: {
          SetFinal(s, s == superfinal_ ? Weight::One() : Weight::Zero());
          break;
        }
      }
    }
    return CacheImpl<B>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<B>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<B>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<B>::NumOutputEpsilons(s);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) && (fst_->Properties(kError, false) ||
                            (mapper_->Properties(0) & kError))) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  void InitArcIterator(StateId s, ArcIteratorData<B> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl<B>::InitArcIterator(s, data);
  }

  void Expand(StateId s) {
    // Add exiting arcs.
    if (s == superfinal_) {
      SetArcs(s);
      return;
    }
    for (ArcIterator<Fst<A>> aiter(*fst_, FindIState(s)); !aiter.Done();
         aiter.Next()) {
      auto aarc = aiter.Value();
      aarc.nextstate = FindOState(aarc.nextstate);
      const auto &barc = (*mapper_)(aarc);
      PushArc(s, barc);
    }

    // Check for superfinal arcs.
    if (!HasFinal(s) || Final(s) == Weight::Zero()) {
      switch (final_action_) {
        case MAP_NO_SUPERFINAL:
        default:
          break;
        case MAP_ALLOW_SUPERFINAL: {
          auto final_arc =
              (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
          if (final_arc.ilabel != 0 || final_arc.olabel != 0) {
            if (superfinal_ == kNoStateId) superfinal_ = nstates_++;
            final_arc.nextstate = superfinal_;
            PushArc(s, final_arc);
          }
          break;
        }
        case MAP_REQUIRE_SUPERFINAL: {
          const auto final_arc =
              (*mapper_)(A(0, 0, fst_->Final(FindIState(s)), kNoStateId));
          if (final_arc.ilabel != 0 || final_arc.olabel != 0 ||
              final_arc.weight != B::Weight::Zero()) {
            PushArc(s, B(final_arc.ilabel, final_arc.olabel, final_arc.weight,
                         superfinal_));
          }
          break;
        }
      }
    }
    SetArcs(s);
  }

 private:
  void Init() {
    SetType("map");
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
    if (fst_->Start() == kNoStateId) {
      final_action_ = MAP_NO_SUPERFINAL;
      SetProperties(kNullProperties);
    } else {
      final_action_ = mapper_->FinalAction();
      uint64 props = fst_->Properties(kCopyProperties, false);
      SetProperties(mapper_->Properties(props));
      if (final_action_ == MAP_REQUIRE_SUPERFINAL) superfinal_ = 0;
    }
  }

  // Maps from output state to input state.
  StateId FindIState(StateId s) {
    if (superfinal_ == kNoStateId || s < superfinal_) {
      return s;
    } else {
      return s - 1;
    }
  }

  // Maps from input state to output state.
  StateId FindOState(StateId is) {
    auto os = is;
    if (!(superfinal_ == kNoStateId || is < superfinal_)) ++os;
    if (os >= nstates_) nstates_ = os + 1;
    return os;
  }

  std::unique_ptr<const Fst<A>> fst_;
  C *mapper_;
  const bool own_mapper_;
  MapFinalAction final_action_;
  StateId superfinal_;
  StateId nstates_;
};

}  // namespace internal

// Maps an arc type A to an arc type B using Mapper function object
// C. This version is a delayed FST.
template <class A, class B, class C>
class ArcMapFst : public ImplToFst<internal::ArcMapFstImpl<A, B, C>> {
 public:
  using Arc = B;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = DefaultCacheStore<B>;
  using State = typename Store::State;
  using Impl = internal::ArcMapFstImpl<A, B, C>;

  friend class ArcIterator<ArcMapFst<A, B, C>>;
  friend class StateIterator<ArcMapFst<A, B, C>>;

  ArcMapFst(const Fst<A> &fst, const C &mapper, const ArcMapFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, mapper, opts)) {}

  ArcMapFst(const Fst<A> &fst, C *mapper, const ArcMapFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, mapper, opts)) {}

  ArcMapFst(const Fst<A> &fst, const C &mapper)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, mapper, ArcMapFstOptions())) {}

  ArcMapFst(const Fst<A> &fst, C *mapper)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, mapper, ArcMapFstOptions())) {}

  // See Fst<>::Copy() for doc.
  ArcMapFst(const ArcMapFst<A, B, C> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this ArcMapFst. See Fst<>::Copy() for further doc.
  ArcMapFst<A, B, C> *Copy(bool safe = false) const override {
    return new ArcMapFst<A, B, C>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<B> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<B> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

 protected:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

 private:
  ArcMapFst &operator=(const ArcMapFst &) = delete;
};

// Specialization for ArcMapFst.
//
// This may be derived from.
template <class A, class B, class C>
class StateIterator<ArcMapFst<A, B, C>> : public StateIteratorBase<B> {
 public:
  using StateId = typename B::StateId;

  explicit StateIterator(const ArcMapFst<A, B, C> &fst)
      : impl_(fst.GetImpl()),
        siter_(*impl_->fst_),
        s_(0),
        superfinal_(impl_->final_action_ == MAP_REQUIRE_SUPERFINAL) {
    CheckSuperfinal();
  }

  bool Done() const final { return siter_.Done() && !superfinal_; }

  StateId Value() const final { return s_; }

  void Next() final {
    ++s_;
    if (!siter_.Done()) {
      siter_.Next();
      CheckSuperfinal();
    } else if (superfinal_) {
      superfinal_ = false;
    }
  }

  void Reset() final {
    s_ = 0;
    siter_.Reset();
    superfinal_ = impl_->final_action_ == MAP_REQUIRE_SUPERFINAL;
    CheckSuperfinal();
  }

 private:
  void CheckSuperfinal() {
    if (impl_->final_action_ != MAP_ALLOW_SUPERFINAL || superfinal_) return;
    if (!siter_.Done()) {
      const auto final_arc =
          (*impl_->mapper_)(A(0, 0, impl_->fst_->Final(s_), kNoStateId));
      if (final_arc.ilabel != 0 || final_arc.olabel != 0) superfinal_ = true;
    }
  }

  const internal::ArcMapFstImpl<A, B, C> *impl_;
  StateIterator<Fst<A>> siter_;
  StateId s_;
  bool superfinal_;  // True if there is a superfinal state and not done.
};

// Specialization for ArcMapFst.
template <class A, class B, class C>
class ArcIterator<ArcMapFst<A, B, C>>
    : public CacheArcIterator<ArcMapFst<A, B, C>> {
 public:
  using StateId = typename A::StateId;

  ArcIterator(const ArcMapFst<A, B, C> &fst, StateId s)
      : CacheArcIterator<ArcMapFst<A, B, C>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class A, class B, class C>
inline void ArcMapFst<A, B, C>::InitStateIterator(
    StateIteratorData<B> *data) const {
  data->base = new StateIterator<ArcMapFst<A, B, C>>(*this);
}

// Utility Mappers.

// Mapper that returns its input.
template <class A>
class IdentityArcMapper {
 public:
  using FromArc = A;
  using ToArc = A;

  ToArc operator()(const FromArc &arc) const { return arc; }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }
};

// Mapper that converts all input symbols to epsilon.
template <class A>
class InputEpsilonMapper {
 public:
  using FromArc = A;
  using ToArc = A;

  ToArc operator()(const FromArc &arc) const {
    return ToArc(0, arc.olabel, arc.weight, arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return (props & kSetArcProperties) | kIEpsilons;
  }
};

// Mapper that converts all output symbols to epsilon.
template <class A>
class OutputEpsilonMapper {
 public:
  using FromArc = A;
  using ToArc = A;

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, 0, arc.weight, arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return (props & kSetArcProperties) | kOEpsilons;
  }
};

// Mapper that returns its input with final states redirected to a single
// super-final state.
template <class A>
class SuperFinalMapper {
 public:
  using FromArc = A;
  using ToArc = A;
  using Label = typename FromArc::Label;
  using Weight = typename FromArc::Weight;;

  // Arg allows setting super-final label.
  explicit SuperFinalMapper(Label final_label = 0)
      : final_label_(final_label) {}

  ToArc operator()(const FromArc &arc) const {
    // Super-final arc.
    if (arc.nextstate == kNoStateId && arc.weight != Weight::Zero()) {
      return ToArc(final_label_, final_label_, arc.weight, kNoStateId);
    } else {
      return arc;
    }
  }

  constexpr MapFinalAction FinalAction() const {
    return MAP_REQUIRE_SUPERFINAL;
  }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    if (final_label_ == 0) {
      return props & kAddSuperFinalProperties;
    } else {
      return props & kAddSuperFinalProperties &
          kILabelInvariantProperties & kOLabelInvariantProperties;
    }
  }

 private:
  Label final_label_;
};

// Mapper that leaves labels and nextstate unchanged and constructs a new weight
// from the underlying value of the arc weight. If no weight converter is
// explictly specified, requires that there is a WeightConvert class
// specialization that converts the weights.
template <class A, class B,
          class C = WeightConvert<typename A::Weight, typename B::Weight>>
class WeightConvertMapper {
 public:
  using FromArc = A;
  using ToArc = B;
  using Converter = C;
  using FromWeight = typename FromArc::Weight;
  using ToWeight = typename ToArc::Weight;

  explicit WeightConvertMapper(const Converter &c = Converter())
      : convert_weight_(c) {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, arc.olabel, convert_weight_(arc.weight),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }

 private:
  Converter convert_weight_;
};

// Non-precision-changing weight conversions; consider using more efficient
// Cast method instead.

using StdToLogMapper = WeightConvertMapper<StdArc, LogArc>;

using LogToStdMapper = WeightConvertMapper<LogArc, StdArc>;

// Precision-changing weight conversions.

using StdToLog64Mapper = WeightConvertMapper<StdArc, Log64Arc>;

using LogToLog64Mapper = WeightConvertMapper<LogArc, Log64Arc>;

using Log64ToStdMapper = WeightConvertMapper<Log64Arc, StdArc>;

using Log64ToLogMapper = WeightConvertMapper<Log64Arc, LogArc>;

// Mapper from A to GallicArc<A>.
template <class A, GallicType G = GALLIC_LEFT>
class ToGallicMapper {
 public:
  using FromArc = A;
  using ToArc = GallicArc<A, G>;

  using SW = StringWeight<typename A::Label, GallicStringType(G)>;
  using AW = typename FromArc::Weight;
  using GW = typename ToArc::Weight;

  ToArc operator()(const FromArc &arc) const {
    // Super-final arc.
    if (arc.nextstate == kNoStateId && arc.weight != AW::Zero()) {
      return ToArc(0, 0, GW(SW::One(), arc.weight), kNoStateId);
    // Super-non-final arc.
    } else if (arc.nextstate == kNoStateId) {
      return ToArc(0, 0, GW::Zero(), kNoStateId);
    // Epsilon label.
    } else if (arc.olabel == 0) {
      return ToArc(arc.ilabel, arc.ilabel, GW(SW::One(), arc.weight),
                   arc.nextstate);
    // Regular label.
    } else {
      return ToArc(arc.ilabel, arc.ilabel, GW(SW(arc.olabel), arc.weight),
                   arc.nextstate);
    }
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return ProjectProperties(props, true) & kWeightInvariantProperties;
  }
};

// Mapper from GallicArc<A> to A.
template <class A, GallicType G = GALLIC_LEFT>
class FromGallicMapper {
 public:
  using FromArc = GallicArc<A, G>;
  using ToArc = A;

  using Label = typename ToArc::Label;
  using AW = typename ToArc::Weight;
  using GW = typename FromArc::Weight;

  explicit FromGallicMapper(Label superfinal_label = 0)
      : superfinal_label_(superfinal_label), error_(false) {}

  ToArc operator()(const FromArc &arc) const {
    // 'Super-non-final' arc.
    if (arc.nextstate == kNoStateId && arc.weight == GW::Zero()) {
      return A(arc.ilabel, 0, AW::Zero(), kNoStateId);
    }
    Label l = kNoLabel;
    AW weight;
    if (!Extract(arc.weight, &weight, &l) || arc.ilabel != arc.olabel) {
      FSTERROR() << "FromGallicMapper: Unrepresentable weight: " << arc.weight
                 << " for arc with ilabel = " << arc.ilabel
                 << ", olabel = " << arc.olabel
                 << ", nextstate = " << arc.nextstate;
      error_ = true;
    }
    if (arc.ilabel == 0 && l != 0 && arc.nextstate == kNoStateId) {
      return ToArc(superfinal_label_, l, weight, arc.nextstate);
    } else {
      return ToArc(arc.ilabel, l, weight, arc.nextstate);
    }
  }

  constexpr MapFinalAction FinalAction() const { return MAP_ALLOW_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 inprops) const {
    uint64 outprops = inprops & kOLabelInvariantProperties &
                      kWeightInvariantProperties & kAddSuperFinalProperties;
    if (error_) outprops |= kError;
    return outprops;
  }

 private:
  template <GallicType GT>
  static bool Extract(const GallicWeight<Label, AW, GT> &gallic_weight,
                      typename A::Weight *weight, typename A::Label *label) {
    using GW = StringWeight<Label, GallicStringType(GT)>;
    const GW &w1 = gallic_weight.Value1();
    const AW &w2 = gallic_weight.Value2();
    typename GW::Iterator iter1(w1);
    const Label l = w1.Size() == 1 ? iter1.Value() : 0;
    if (l == kStringInfinity || l == kStringBad || w1.Size() > 1) return false;
    *label = l;
    *weight = w2;
    return true;
  }

  static bool Extract(const GallicWeight<Label, AW, GALLIC> &gallic_weight,
                      typename A::Weight *weight, typename A::Label *label) {
    if (gallic_weight.Size() > 1) return false;
    if (gallic_weight.Size() == 0) {
      *label = 0;
      *weight = A::Weight::Zero();
      return true;
    }
    return Extract<GALLIC_RESTRICT>(gallic_weight.Back(), weight, label);
  }

  const Label superfinal_label_;
  mutable bool error_;
};

// Mapper from GallicArc<A> to A.
template <class A, GallicType G = GALLIC_LEFT>
class GallicToNewSymbolsMapper {
 public:
  using FromArc = GallicArc<A, G>;
  using ToArc = A;

  using Label = typename ToArc::Label;
  using StateId = typename ToArc::StateId;
  using AW = typename ToArc::Weight;
  using GW = typename FromArc::Weight;
  using SW = StringWeight<Label, GallicStringType(G)>;

  explicit GallicToNewSymbolsMapper(MutableFst<ToArc> *fst)
      : fst_(fst),
        lmax_(0),
        osymbols_(fst->OutputSymbols()),
        isymbols_(nullptr),
        error_(false) {
    fst_->DeleteStates();
    state_ = fst_->AddState();
    fst_->SetStart(state_);
    fst_->SetFinal(state_, AW::One());
    if (osymbols_) {
      string name = osymbols_->Name() + "_from_gallic";
      fst_->SetInputSymbols(new SymbolTable(name));
      isymbols_ = fst_->MutableInputSymbols();
      const int64 zero = 0;
      isymbols_->AddSymbol(osymbols_->Find(zero), 0);
    } else {
      fst_->SetInputSymbols(nullptr);
    }
  }

  ToArc operator()(const FromArc &arc) {
    // Super-non-final arc.
    if (arc.nextstate == kNoStateId && arc.weight == GW::Zero()) {
      return ToArc(arc.ilabel, 0, AW::Zero(), kNoStateId);
    }
    SW w1 = arc.weight.Value1();
    AW w2 = arc.weight.Value2();
    Label l;
    if (w1.Size() == 0) {
      l = 0;
    } else {
      auto insert_result = map_.insert(std::make_pair(w1, kNoLabel));
      if (!insert_result.second) {
        l = insert_result.first->second;
      } else {
        l = ++lmax_;
        insert_result.first->second = l;
        StringWeightIterator<SW> iter1(w1);
        StateId n;
        string s;
        for (size_t i = 0, p = state_; i < w1.Size();
             ++i, iter1.Next(), p = n) {
          n = i == w1.Size() - 1 ? state_ : fst_->AddState();
          fst_->AddArc(p, ToArc(i ? 0 : l, iter1.Value(), AW::One(), n));
          if (isymbols_) {
            if (i) s = s + "_";
            s = s + osymbols_->Find(iter1.Value());
          }
        }
        if (isymbols_) isymbols_->AddSymbol(s, l);
      }
    }
    if (l == kStringInfinity || l == kStringBad || arc.ilabel != arc.olabel) {
      FSTERROR() << "GallicToNewSymbolMapper: Unrepresentable weight: " << l;
      error_ = true;
    }
    return ToArc(arc.ilabel, l, w2, arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_ALLOW_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 inprops) const {
    uint64 outprops = inprops & kOLabelInvariantProperties &
                      kWeightInvariantProperties & kAddSuperFinalProperties;
    if (error_) outprops |= kError;
    return outprops;
  }

 private:
  class StringKey {
   public:
    size_t operator()(const SW &x) const { return x.Hash(); }
  };

  using Map = std::unordered_map<SW, Label, StringKey>;

  MutableFst<ToArc> *fst_;
  Map map_;
  Label lmax_;
  StateId state_;
  const SymbolTable *osymbols_;
  SymbolTable *isymbols_;
  mutable bool error_;
};

// TODO(kbg): Add common base class for those mappers which do nothing except
// mutate their weights.

// Mapper to add a constant to all weights.
template <class A>
class PlusMapper {
 public:
  using FromArc = A;
  using ToArc = A;
  using Weight = typename FromArc::Weight;

  explicit PlusMapper(Weight weight) : weight_(std::move(weight)) {}

  ToArc operator()(const FromArc &arc) const {
    if (arc.weight == Weight::Zero()) return arc;
    return ToArc(arc.ilabel, arc.olabel, Plus(arc.weight, weight_),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kWeightInvariantProperties;
  }

 private:
  const Weight weight_;
};

// Mapper to (right) multiply a constant to all weights.
template <class A>
class TimesMapper {
 public:
  using FromArc = A;
  using ToArc = A;
  using Weight = typename FromArc::Weight;

  explicit TimesMapper(Weight weight) : weight_(std::move(weight)) {}

  ToArc operator()(const FromArc &arc) const {
    if (arc.weight == Weight::Zero()) return arc;
    return ToArc(arc.ilabel, arc.olabel, Times(arc.weight, weight_),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kWeightInvariantProperties;
  }

 private:
  const Weight weight_;
};

// Mapper to take all weights to a constant power. The power argument is stored
// as a double, so if there is a floating-point power implementation for this
// weight type, it will take precedence. Otherwise, the power argument's 53 bits
// of integer precision will be implicitly converted to a size_t and the default
// power implementation (iterated multiplication) will be used instead.
template <class A>
class PowerMapper {
 public:
  using FromArc = A;
  using ToArc = A;
  using Weight = typename FromArc::Weight;

  explicit PowerMapper(double power) : power_(power) {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, arc.olabel, Power(arc.weight, power_),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kWeightInvariantProperties;
  }

 private:
  const double power_;
};

// Mapper to reciprocate all non-Zero() weights.
template <class A>
class InvertWeightMapper {
 public:
  using FromArc = A;
  using ToArc = A;
  using Weight = typename FromArc::Weight;

  ToArc operator()(const FromArc &arc) const {
    if (arc.weight == Weight::Zero()) return arc;
    return ToArc(arc.ilabel, arc.olabel, Divide(Weight::One(), arc.weight),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kWeightInvariantProperties;
  }
};

// Mapper to map all non-Zero() weights to One().
template <class A, class B = A>
class RmWeightMapper {
 public:
  using FromArc = A;
  using ToArc = B;
  using FromWeight = typename FromArc::Weight;
  using ToWeight = typename ToArc::Weight;

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, arc.olabel,
                 arc.weight != FromWeight::Zero() ?
                 ToWeight::One() : ToWeight::Zero(),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return (props & kWeightInvariantProperties) | kUnweighted;
  }
};

// Mapper to quantize all weights.
template <class A, class B = A>
class QuantizeMapper {
 public:
  using FromArc = A;
  using ToArc = B;
  using FromWeight = typename FromArc::Weight;
  using ToWeight = typename ToArc::Weight;

  QuantizeMapper() : delta_(kDelta) {}

  explicit QuantizeMapper(float d) : delta_(d) {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, arc.olabel, arc.weight.Quantize(delta_),
                 arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props & kWeightInvariantProperties;
  }

 private:
  const float delta_;
};

// Mapper from A to B under the assumption:
//
//    B::Weight = A::Weight::ReverseWeight
//    B::Label == A::Label
//    B::StateId == A::StateId
//
// The weight is reversed, while the label and nextstate are preserved.
template <class A, class B>
class ReverseWeightMapper {
 public:
  using FromArc = A;
  using ToArc = B;

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, arc.olabel, arc.weight.Reverse(), arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }
};

}  // namespace fst

#endif  // FST_ARC_MAP_H_
