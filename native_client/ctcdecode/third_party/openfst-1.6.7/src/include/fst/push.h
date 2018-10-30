// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to reweight/push an FST, and utility functions to weigh and reweight
// an FST.

#ifndef FST_PUSH_H_
#define FST_PUSH_H_

#include <vector>

#include <fst/log.h>

#include <fst/arc-map.h>
#include <fst/factor-weight.h>
#include <fst/fst.h>
#include <fst/reweight.h>
#include <fst/shortest-distance.h>


namespace fst {

// Computes the total weight (sum of the weights of all accepting paths) from
// the output of ShortestDistance, using the shortest distance from the final
// state when reverse is true and from the initial state otherwise.
template <class Arc>
typename Arc::Weight ComputeTotalWeight(
    const Fst<Arc> &fst, const std::vector<typename Arc::Weight> &distance,
    bool reverse) {
  if (reverse) {
    return fst.Start() < distance.size() ? distance[fst.Start()]
                                         : Arc::Weight::Zero();
  }
  auto sum = Arc::Weight::Zero();
  for (typename Arc::StateId s = 0; s < distance.size(); ++s) {
    sum = Plus(sum, Times(distance[s], fst.Final(s)));
  }
  return sum;
}

// Divides the weight of every accepting path by a fixed weight. This weight
// is also divided at the final state if at_final is true and at the initial
// state otherwise.
template <class Arc>
void RemoveWeight(MutableFst<Arc> *fst, const typename Arc::Weight &weight,
                  bool at_final) {
  using Weight = typename Arc::Weight;
  if ((weight == Weight::One()) || (weight == Weight::Zero())) return;
  if (at_final) {
    for (StateIterator<MutableFst<Arc>> siter(*fst); !siter.Done();
         siter.Next()) {
      fst->SetFinal(siter.Value(),
                    Divide(fst->Final(siter.Value()), weight, DIVIDE_RIGHT));
    }
  } else {
    const auto start = fst->Start();
    for (MutableArcIterator<MutableFst<Arc>> aiter(fst, start); !aiter.Done();
         aiter.Next()) {
      auto arc = aiter.Value();
      arc.weight = Divide(arc.weight, weight, DIVIDE_LEFT);
      aiter.SetValue(arc);
    }
    fst->SetFinal(start, Divide(fst->Final(start), weight, DIVIDE_LEFT));
  }
}

// Pushes the weights in FST in the direction defined by TYPE. If
// pushing towards the initial state, the sum of the weight of the
// outgoing transitions and final weight at a non-initial state is
// equal to One() in the resulting machine. If pushing towards the
// final state, the same property holds on the reverse machine.
//
// Weight needs to be left distributive when pushing towards the
// initial state and right distributive when pushing towards the final
// states.
template <class Arc>
void Push(MutableFst<Arc> *fst, ReweightType type, float delta = kDelta,
          bool remove_total_weight = false) {
  using Weight = typename Arc::Weight;
  std::vector<Weight> distance;
  ShortestDistance(*fst, &distance, type == REWEIGHT_TO_INITIAL, delta);
  auto total_weight = Weight::One();
  if (remove_total_weight) {
    total_weight =
        ComputeTotalWeight(*fst, distance, type == REWEIGHT_TO_INITIAL);
  }
  Reweight(fst, distance, type);
  if (remove_total_weight) {
    RemoveWeight(fst, total_weight, type == REWEIGHT_TO_FINAL);
  }
}

constexpr uint32 kPushWeights = 0x0001;
constexpr uint32 kPushLabels = 0x0002;
constexpr uint32 kPushRemoveTotalWeight = 0x0004;
constexpr uint32 kPushRemoveCommonAffix = 0x0008;

// Pushes the weights and/or labels of the input FST into the output
// mutable FST by pushing weights and/or labels (as determined by the
// ptype argument) towards the initial state or final states (as
// determined by the rtype template parameter). The weight type must
// be left distributive when pushing weights towards the initial state, and
// right distribution when pushing weights towards the final states.
template <class Arc, ReweightType rtype>
void Push(const Fst<Arc> &ifst, MutableFst<Arc> *ofst, uint32 ptype,
          float delta = kDelta) {
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;
  if ((ptype & (kPushWeights | kPushLabels)) == kPushWeights) {
    *ofst = ifst;
    Push(ofst, rtype, delta, ptype & kPushRemoveTotalWeight);
  } else if (ptype & kPushLabels) {
    const auto gtype =
        rtype == REWEIGHT_TO_INITIAL ? GALLIC_LEFT : GALLIC_RIGHT;
    using GallicWeight = typename GallicArc<Arc, gtype>::Weight;
    std::vector<GallicWeight> gdistance;
    VectorFst<GallicArc<Arc, gtype>> gfst;
    ArcMap(ifst, &gfst, ToGallicMapper<Arc, gtype>());
    if (ptype & kPushWeights) {
      ShortestDistance(gfst, &gdistance, rtype == REWEIGHT_TO_INITIAL, delta);
    } else {
      ArcMapFst<Arc, Arc, RmWeightMapper<Arc>> uwfst(ifst,
                                                      RmWeightMapper<Arc>());
      ArcMapFst<Arc, GallicArc<Arc, gtype>, ToGallicMapper<Arc, gtype>> guwfst(
          uwfst, ToGallicMapper<Arc, gtype>());
      ShortestDistance(guwfst, &gdistance, rtype == REWEIGHT_TO_INITIAL, delta);
    }
    auto total_weight = GallicWeight::One();
    if (ptype & (kPushRemoveTotalWeight | kPushRemoveCommonAffix)) {
      total_weight =
          ComputeTotalWeight(gfst, gdistance, rtype == REWEIGHT_TO_INITIAL);
      total_weight = GallicWeight(
          ptype & kPushRemoveCommonAffix
              ? total_weight.Value1()
              : StringWeight<Label, GallicStringType(gtype)>::One(),
          ptype & kPushRemoveTotalWeight ? total_weight.Value2()
                                         : Weight::One());
    }
    Reweight(&gfst, gdistance, rtype);
    if (ptype & (kPushRemoveTotalWeight | kPushRemoveCommonAffix)) {
      RemoveWeight(&gfst, total_weight, rtype == REWEIGHT_TO_FINAL);
    }
    FactorWeightFst<GallicArc<Arc, gtype>, GallicFactor<Label, Weight, gtype>>
        fwfst(gfst);
    ArcMap(fwfst, ofst, FromGallicMapper<Arc, gtype>());
    ofst->SetOutputSymbols(ifst.OutputSymbols());
  } else {
    LOG(WARNING) << "Push: pushing type is set to 0, so not pushing";
    *ofst = ifst;
  }
}

}  // namespace fst

#endif  // FST_PUSH_H_
