// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to compute the concatenation of two FSTs.

#ifndef FST_CONCAT_H_
#define FST_CONCAT_H_

#include <algorithm>
#include <vector>

#include <fst/mutable-fst.h>
#include <fst/rational.h>


namespace fst {

// Computes the concatenation (product) of two FSTs. If FST1 transduces string
// x to y with weight a and FST2 transduces string w to v with weight b, then
// their concatenation transduces string xw to yv with weight Times(a, b).
//
// This version modifies its MutableFst argument (in first position).
//
// Complexity:
//
//   Time: O(V1 + V2 + E2)
//   Space: O(V1 + V2 + E2)
//
// where Vi is the number of states, and Ei is the number of arcs, of the ith
// FST.
template <class Arc>
void Concat(MutableFst<Arc> *fst1, const Fst<Arc> &fst2) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  // Checks that the symbol table are compatible.
  if (!CompatSymbols(fst1->InputSymbols(), fst2.InputSymbols()) ||
      !CompatSymbols(fst1->OutputSymbols(), fst2.OutputSymbols())) {
    FSTERROR() << "Concat: Input/output symbol tables of 1st argument "
               << "does not match input/output symbol tables of 2nd argument";
    fst1->SetProperties(kError, kError);
    return;
  }
  const auto props1 = fst1->Properties(kFstProperties, false);
  const auto props2 = fst2.Properties(kFstProperties, false);
  const auto start1 = fst1->Start();
  if (start1 == kNoStateId) {
    if (props2 & kError) fst1->SetProperties(kError, kError);
    return;
  }
  const auto numstates1 = fst1->NumStates();
  if (fst2.Properties(kExpanded, false)) {
    fst1->ReserveStates(numstates1 + CountStates(fst2));
  }
  for (StateIterator<Fst<Arc>> siter2(fst2); !siter2.Done(); siter2.Next()) {
    const auto s1 = fst1->AddState();
    const auto s2 = siter2.Value();
    fst1->SetFinal(s1, fst2.Final(s2));
    fst1->ReserveArcs(s1, fst2.NumArcs(s2));
    for (ArcIterator<Fst<Arc>> aiter(fst2, s2); !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      arc.nextstate += numstates1;
      fst1->AddArc(s1, arc);
    }
  }
  const auto start2 = fst2.Start();
  for (StateId s1 = 0; s1 < numstates1; ++s1) {
    const auto weight = fst1->Final(s1);
    if (weight != Weight::Zero()) {
      fst1->SetFinal(s1, Weight::Zero());
      if (start2 != kNoStateId) {
        fst1->AddArc(s1, Arc(0, 0, weight, start2 + numstates1));
      }
    }
  }
  if (start2 != kNoStateId) {
    fst1->SetProperties(ConcatProperties(props1, props2), kFstProperties);
  }
}

// Computes the concatentation of two FSTs.  This version modifies its
// MutableFst argument (in second position).
//
// Complexity:
//
//   Time: O(V1 + E1)
//   Space: O(V1 + E1)
//
// where Vi is the number of states, and Ei is the number of arcs, of the ith
// FST.
template <class Arc>
void Concat(const Fst<Arc> &fst1, MutableFst<Arc> *fst2) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  // Checks that the symbol table are compatible.
  if (!CompatSymbols(fst1.InputSymbols(), fst2->InputSymbols()) ||
      !CompatSymbols(fst1.OutputSymbols(), fst2->OutputSymbols())) {
    FSTERROR() << "Concat: Input/output symbol tables of 1st argument "
               << "does not match input/output symbol tables of 2nd argument";
    fst2->SetProperties(kError, kError);
    return;
  }
  const auto props1 = fst1.Properties(kFstProperties, false);
  const auto props2 = fst2->Properties(kFstProperties, false);
  const auto start2 = fst2->Start();
  if (start2 == kNoStateId) {
    if (props1 & kError) fst2->SetProperties(kError, kError);
    return;
  }
  const auto numstates2 = fst2->NumStates();
  if (fst1.Properties(kExpanded, false)) {
    fst2->ReserveStates(numstates2 + CountStates(fst1));
  }
  for (StateIterator<Fst<Arc>> siter(fst1); !siter.Done(); siter.Next()) {
    const auto s1 = siter.Value();
    const auto s2 = fst2->AddState();
    const auto weight = fst1.Final(s1);
    if (weight != Weight::Zero()) {
      fst2->ReserveArcs(s2, fst1.NumArcs(s1) + 1);
      fst2->AddArc(s2, Arc(0, 0, weight, start2));
    } else {
      fst2->ReserveArcs(s2, fst1.NumArcs(s1));
    }
    for (ArcIterator<Fst<Arc>> aiter(fst1, s1); !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      arc.nextstate += numstates2;
      fst2->AddArc(s2, arc);
    }
  }
  const auto start1 = fst1.Start();
  if (start1 != kNoStateId) {
    fst2->SetStart(start1 + numstates2);
    fst2->SetProperties(ConcatProperties(props1, props2), kFstProperties);
  } else {
    fst2->SetStart(fst2->AddState());
  }
}

// Computes the concatentation of two FSTs. This version modifies its
// RationalFst input (in first position).
template <class Arc>
void Concat(RationalFst<Arc> *fst1, const Fst<Arc> &fst2) {
  fst1->GetMutableImpl()->AddConcat(fst2, true);
}

// Computes the concatentation of two FSTs. This version modifies its
// RationalFst input (in second position).
template <class Arc>
void Concat(const Fst<Arc> &fst1, RationalFst<Arc> *fst2) {
  fst2->GetMutableImpl()->AddConcat(fst1, false);
}

using ConcatFstOptions = RationalFstOptions;

// Computes the concatenation (product) of two FSTs; this version is a delayed
// FST. If FST1 transduces string x to y with weight a and FST2 transduces
// string w to v with weight b, then their concatenation transduces string xw
// to yv with Times(a, b).
//
// Complexity:
//
//   Time: O(v1 + e1 + v2 + e2),
//   Space: O(v1 + v2)
//
// where vi is the number of states visited, and ei is the number of arcs
// visited, of the ith FST. Constant time and space to visit an input state or
// arc is assumed and exclusive of caching.
template <class A>
class ConcatFst : public RationalFst<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  ConcatFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {
    GetMutableImpl()->InitConcat(fst1, fst2);
  }

  ConcatFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
            const ConcatFstOptions &opts)
      : RationalFst<Arc>(opts) {
    GetMutableImpl()->InitConcat(fst1, fst2);
  }

  // See Fst<>::Copy() for doc.
  ConcatFst(const ConcatFst<Arc> &fst, bool safe = false)
      : RationalFst<Arc>(fst, safe) {}

  // Get a copy of this ConcatFst. See Fst<>::Copy() for further doc.
  ConcatFst<Arc> *Copy(bool safe = false) const override {
    return new ConcatFst<Arc>(*this, safe);
  }

 private:
  using ImplToFst<internal::RationalFstImpl<Arc>>::GetImpl;
  using ImplToFst<internal::RationalFstImpl<Arc>>::GetMutableImpl;
};

// Specialization for ConcatFst.
template <class Arc>
class StateIterator<ConcatFst<Arc>> : public StateIterator<RationalFst<Arc>> {
 public:
  explicit StateIterator(const ConcatFst<Arc> &fst)
      : StateIterator<RationalFst<Arc>>(fst) {}
};

// Specialization for ConcatFst.
template <class Arc>
class ArcIterator<ConcatFst<Arc>> : public ArcIterator<RationalFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const ConcatFst<Arc> &fst, StateId s)
      : ArcIterator<RationalFst<Arc>>(fst, s) {}
};

// Useful alias when using StdArc.
using StdConcatFst = ConcatFst<StdArc>;

}  // namespace fst

#endif  // FST_CONCAT_H_
