// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to compute the union of two FSTs.

#ifndef FST_UNION_H_
#define FST_UNION_H_

#include <algorithm>
#include <vector>

#include <fst/mutable-fst.h>
#include <fst/rational.h>


namespace fst {

// Computes the union (sum) of two FSTs. This version writes the union to an
// output MutableFst. If A transduces string x to y with weight a and B
// transduces string w to v with weight b, then their union transduces x to y
// with weight a and w to v with weight b.
//
// Complexity:
//
//   Time: (V_2 + E_2)
//   Space: O(V_2 + E_2)
//
// where Vi is the number of states, and Ei is the number of arcs, in the ith
// FST.
template <class Arc>
void Union(MutableFst<Arc> *fst1, const Fst<Arc> &fst2) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  // Checks for symbol table compatibility.
  if (!CompatSymbols(fst1->InputSymbols(), fst2.InputSymbols()) ||
      !CompatSymbols(fst1->OutputSymbols(), fst2.OutputSymbols())) {
    FSTERROR() << "Union: Input/output symbol tables of 1st argument "
               << "do not match input/output symbol tables of 2nd argument";
    fst1->SetProperties(kError, kError);
    return;
  }
  const auto numstates1 = fst1->NumStates();
  const bool initial_acyclic1 = fst1->Properties(kInitialAcyclic, true);
  const auto props1 = fst1->Properties(kFstProperties, false);
  const auto props2 = fst2.Properties(kFstProperties, false);
  const auto start2 = fst2.Start();
  if (start2 == kNoStateId) {
    if (props2 & kError) fst1->SetProperties(kError, kError);
    return;
  }
  if (fst2.Properties(kExpanded, false)) {
    fst1->ReserveStates(numstates1 + CountStates(fst2) +
                        (initial_acyclic1 ? 0 : 1));
  }
  for (StateIterator<Fst<Arc>> siter(fst2); !siter.Done(); siter.Next()) {
    const auto s1 = fst1->AddState();
    const auto s2 = siter.Value();
    fst1->SetFinal(s1, fst2.Final(s2));
    fst1->ReserveArcs(s1, fst2.NumArcs(s2));
    for (ArcIterator<Fst<Arc>> aiter(fst2, s2); !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();  // Copy intended.
      arc.nextstate += numstates1;
      fst1->AddArc(s1, arc);
    }
  }
  const auto start1 = fst1->Start();
  if (start1 == kNoStateId) {
    fst1->SetStart(start2);
    fst1->SetProperties(props2, kCopyProperties);
    return;
  }
  if (initial_acyclic1) {
    fst1->AddArc(start1, Arc(0, 0, Weight::One(), start2 + numstates1));
  } else {
    const auto nstart1 = fst1->AddState();
    fst1->SetStart(nstart1);
    fst1->AddArc(nstart1, Arc(0, 0, Weight::One(), start1));
    fst1->AddArc(nstart1, Arc(0, 0, Weight::One(), start2 + numstates1));
  }
  fst1->SetProperties(UnionProperties(props1, props2), kFstProperties);
}

// Computes the union of two FSTs, modifying the RationalFst argument.
template <class Arc>
void Union(RationalFst<Arc> *fst1, const Fst<Arc> &fst2) {
  fst1->GetMutableImpl()->AddUnion(fst2);
}

using UnionFstOptions = RationalFstOptions;

// Computes the union (sum) of two FSTs. This version is a delayed FST. If A
// transduces string x to y with weight a and B transduces string w to v with
// weight b, then their union transduces x to y with weight a and w to v with
// weight b.
//
// Complexity:
//
//   Time: O(v_1 + e_1 + v_2 + e_2)
//   Space: O(v_1 + v_2)
//
// where vi is the number of states visited, and ei is the number of arcs
// visited, in the ith FST. Constant time and space to visit an input state or
// arc is assumed and exclusive of caching.
template <class A>
class UnionFst : public RationalFst<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  UnionFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {
    GetMutableImpl()->InitUnion(fst1, fst2);
  }

  UnionFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
           const UnionFstOptions &opts)
      : RationalFst<Arc>(opts) {
    GetMutableImpl()->InitUnion(fst1, fst2);
  }

  // See Fst<>::Copy() for doc.
  UnionFst(const UnionFst<Arc> &fst, bool safe = false)
      : RationalFst<Arc>(fst, safe) {}

  // Gets a copy of this UnionFst. See Fst<>::Copy() for further doc.
  UnionFst<Arc> *Copy(bool safe = false) const override {
    return new UnionFst<Arc>(*this, safe);
  }

 private:
  using ImplToFst<internal::RationalFstImpl<Arc>>::GetImpl;
  using ImplToFst<internal::RationalFstImpl<Arc>>::GetMutableImpl;
};

// Specialization for UnionFst.
template <class Arc>
class StateIterator<UnionFst<Arc>> : public StateIterator<RationalFst<Arc>> {
 public:
  explicit StateIterator(const UnionFst<Arc> &fst)
      : StateIterator<RationalFst<Arc>>(fst) {}
};

// Specialization for UnionFst.
template <class Arc>
class ArcIterator<UnionFst<Arc>> : public ArcIterator<RationalFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const UnionFst<Arc> &fst, StateId s)
      : ArcIterator<RationalFst<Arc>>(fst, s) {}
};

using StdUnionFst = UnionFst<StdArc>;

}  // namespace fst

#endif  // FST_UNION_H_
