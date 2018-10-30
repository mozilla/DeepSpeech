// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PRUNE_H_
#define FST_SCRIPT_PRUNE_H_

#include <tuple>
#include <utility>

#include <fst/prune.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

using PruneArgs1 = std::tuple<const FstClass &, MutableFstClass *,
                              const WeightClass &, int64, float>;

template <class Arc>
void Prune(PruneArgs1 *args) {
  using Weight = typename Arc::Weight;
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  const auto weight_threshold = *(std::get<2>(*args).GetWeight<Weight>());
  Prune(ifst, ofst, weight_threshold, std::get<3>(*args), std::get<4>(*args));
}

using PruneArgs2 = std::tuple<MutableFstClass *, const WeightClass &, int64,
                               float>;

template <class Arc>
void Prune(PruneArgs2 *args) {
  using Weight = typename Arc::Weight;
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  const auto weight_threshold = *(std::get<1>(*args).GetWeight<Weight>());
  Prune(fst, weight_threshold, std::get<2>(*args), std::get<3>(*args));
}

void Prune(const FstClass &ifst, MutableFstClass *ofst,
           const WeightClass &weight_threshold,
           int64 state_threshold = kNoStateId,
           float delta = kDelta);

void Prune(MutableFstClass *fst, const WeightClass &weight_threshold,
           int64 state_threshold = kNoStateId, float delta = kDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PRUNE_H_
