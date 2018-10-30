// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_INTERSECT_H_
#define FST_SCRIPT_INTERSECT_H_

#include <tuple>

#include <fst/intersect.h>
#include <fst/script/compose.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using IntersectArgs = std::tuple<const FstClass &, const FstClass &,
                                 MutableFstClass *, const ComposeOptions &>;

template <class Arc>
void Intersect(IntersectArgs *args) {
  const Fst<Arc> &ifst1 = *(std::get<0>(*args).GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(std::get<1>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<2>(*args)->GetMutableFst<Arc>();
  const auto &opts = std::get<3>(*args);
  Intersect(ifst1, ifst2, ofst, opts);
}

void Intersect(const FstClass &ifst, const FstClass &ifst2,
               MutableFstClass *ofst,
               const ComposeOptions &opts = ComposeOptions());

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_INTERSECT_H_
