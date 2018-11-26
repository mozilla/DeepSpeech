// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_REWEIGHT_H_
#define FST_SCRIPT_REWEIGHT_H_

#include <tuple>
#include <vector>

#include <fst/reweight.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

using ReweightArgs = std::tuple<MutableFstClass *,
                                const std::vector<WeightClass> &, ReweightType>;

template <class Arc>
void Reweight(ReweightArgs *args) {
  using Weight = typename Arc::Weight;
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  const std::vector<WeightClass> &potentials = std::get<1>(*args);
  std::vector<Weight> typed_potentials;
  internal::CopyWeights(potentials, &typed_potentials);
  Reweight(fst, typed_potentials, std::get<2>(*args));
}

void Reweight(MutableFstClass *fst, const std::vector<WeightClass> &potentials,
              ReweightType reweight_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_REWEIGHT_H_
