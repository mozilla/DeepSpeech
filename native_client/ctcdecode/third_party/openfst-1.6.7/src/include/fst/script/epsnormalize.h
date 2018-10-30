// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_EPSNORMALIZE_H_
#define FST_SCRIPT_EPSNORMALIZE_H_

#include <tuple>

#include <fst/epsnormalize.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using EpsNormalizeArgs = std::tuple<const FstClass &, MutableFstClass *,
                                    EpsNormalizeType>;

template <class Arc>
void EpsNormalize(EpsNormalizeArgs *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  EpsNormalize(ifst, ofst, std::get<2>(*args));
}

void EpsNormalize(const FstClass &ifst, MutableFstClass *ofst,
                  EpsNormalizeType norm_type = EPS_NORM_INPUT);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_EPSNORMALIZE_H_
