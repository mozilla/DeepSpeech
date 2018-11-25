// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Function to reweight an FST.

#ifndef FST_REWEIGHT_H_
#define FST_REWEIGHT_H_

#include <vector>
#include <fst/log.h>

#include <fst/mutable-fst.h>


namespace fst {

enum ReweightType { REWEIGHT_TO_INITIAL, REWEIGHT_TO_FINAL };

// Reweights an FST according to a vector of potentials in a given direction.
// The weight must be left distributive when reweighting towards the initial
// state and right distributive when reweighting towards the final states.
//
// An arc of weight w, with an origin state of potential p and destination state
// of potential q, is reweighted by p^-1 \otimes (w \otimes q) when reweighting
// torwards the initial state, and by (p \otimes w) \otimes q^-1 when
// reweighting towards the final states.
template <class Arc>
void Reweight(MutableFst<Arc> *fst,
              const std::vector<typename Arc::Weight> &potential,
              ReweightType type) {
  using Weight = typename Arc::Weight;
  if (fst->NumStates() == 0) return;
  // TODO(kbg): Make this a compile-time static_assert once we have a pleasant
  // way to "deregister" this operation for non-distributive semirings so an
  // informative error message is produced.
  if (type == REWEIGHT_TO_FINAL && !(Weight::Properties() & kRightSemiring)) {
    FSTERROR() << "Reweight: Reweighting to the final states requires "
               << "Weight to be right distributive: " << Weight::Type();
    fst->SetProperties(kError, kError);
    return;
  }
  // TODO(kbg): Make this a compile-time static_assert once we have a pleasant
  // way to "deregister" this operation for non-distributive semirings so an
  // informative error message is produced.
  if (type == REWEIGHT_TO_INITIAL && !(Weight::Properties() & kLeftSemiring)) {
    FSTERROR() << "Reweight: Reweighting to the initial state requires "
               << "Weight to be left distributive: " << Weight::Type();
    fst->SetProperties(kError, kError);
    return;
  }
  StateIterator<MutableFst<Arc>> siter(*fst);
  for (; !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    if (s == potential.size()) break;
    const auto &weight = potential[s];
    if (weight != Weight::Zero()) {
      for (MutableArcIterator<MutableFst<Arc>> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        auto arc = aiter.Value();
        if (arc.nextstate >= potential.size()) continue;
        const auto &nextweight = potential[arc.nextstate];
        if (nextweight == Weight::Zero()) continue;
        if (type == REWEIGHT_TO_INITIAL) {
          arc.weight =
              Divide(Times(arc.weight, nextweight), weight, DIVIDE_LEFT);
        }
        if (type == REWEIGHT_TO_FINAL) {
          arc.weight =
              Divide(Times(weight, arc.weight), nextweight, DIVIDE_RIGHT);
        }
        aiter.SetValue(arc);
      }
      if (type == REWEIGHT_TO_INITIAL) {
        fst->SetFinal(s, Divide(fst->Final(s), weight, DIVIDE_LEFT));
      }
    }
    if (type == REWEIGHT_TO_FINAL) {
      fst->SetFinal(s, Times(weight, fst->Final(s)));
    }
  }
  // This handles elements past the end of the potentials array.
  for (; !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    if (type == REWEIGHT_TO_FINAL) {
      fst->SetFinal(s, Times(Weight::Zero(), fst->Final(s)));
    }
  }
  const auto startweight = fst->Start() < potential.size()
                               ? potential[fst->Start()]
                               : Weight::Zero();
  if ((startweight != Weight::One()) && (startweight != Weight::Zero())) {
    if (fst->Properties(kInitialAcyclic, true) & kInitialAcyclic) {
      const auto s = fst->Start();
      for (MutableArcIterator<MutableFst<Arc>> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        auto arc = aiter.Value();
        if (type == REWEIGHT_TO_INITIAL) {
          arc.weight = Times(startweight, arc.weight);
        } else {
          arc.weight = Times(Divide(Weight::One(), startweight, DIVIDE_RIGHT),
                             arc.weight);
        }
        aiter.SetValue(arc);
      }
      if (type == REWEIGHT_TO_INITIAL) {
        fst->SetFinal(s, Times(startweight, fst->Final(s)));
      } else {
        fst->SetFinal(s, Times(Divide(Weight::One(), startweight, DIVIDE_RIGHT),
                               fst->Final(s)));
      }
    } else {
      const auto s = fst->AddState();
      const auto weight =
          (type == REWEIGHT_TO_INITIAL)
              ? startweight
              : Divide(Weight::One(), startweight, DIVIDE_RIGHT);
      fst->AddArc(s, Arc(0, 0, weight, fst->Start()));
      fst->SetStart(s);
    }
  }
  fst->SetProperties(ReweightProperties(fst->Properties(kFstProperties, false)),
                     kFstProperties);
}

}  // namespace fst

#endif  // FST_REWEIGHT_H_
