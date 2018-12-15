// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_EQUIVALENT_H_
#define FST_SCRIPT_EQUIVALENT_H_

#include <tuple>

#include <fst/equivalent.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using EquivalentInnerArgs = std::tuple<const FstClass &, const FstClass &,
                                       float>;

using EquivalentArgs = WithReturnValue<bool, EquivalentInnerArgs>;

template <class Arc>
void Equivalent(EquivalentArgs *args) {
  const Fst<Arc> &fst1 = *(std::get<0>(args->args).GetFst<Arc>());
  const Fst<Arc> &fst2 = *(std::get<1>(args->args).GetFst<Arc>());
  args->retval = Equivalent(fst1, fst2, std::get<2>(args->args));
}

bool Equivalent(const FstClass &fst1, const FstClass &fst2,
                float delta = kDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_EQUIVALENT_H_
