// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ISOMORPHIC_H_
#define FST_SCRIPT_ISOMORPHIC_H_

#include <tuple>

#include <fst/isomorphic.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using IsomorphicInnerArgs = std::tuple<const FstClass &, const FstClass &,
                                       float>;

using IsomorphicArgs = WithReturnValue<bool, IsomorphicInnerArgs>;

template <class Arc>
void Isomorphic(IsomorphicArgs *args) {
  const Fst<Arc> &fst1 = *(std::get<0>(args->args).GetFst<Arc>());
  const Fst<Arc> &fst2 = *(std::get<1>(args->args).GetFst<Arc>());
  args->retval = Isomorphic(fst1, fst2, std::get<2>(args->args));
}

bool Isomorphic(const FstClass &fst1, const FstClass &fst2,
                float delta = kDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ISOMORPHIC_H_
