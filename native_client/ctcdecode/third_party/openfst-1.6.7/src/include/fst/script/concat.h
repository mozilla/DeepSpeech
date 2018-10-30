// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CONCAT_H_
#define FST_SCRIPT_CONCAT_H_

#include <utility>

#include <fst/concat.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ConcatArgs1 = std::pair<MutableFstClass *, const FstClass &>;

template <class Arc>
void Concat(ConcatArgs1 *args) {
  MutableFst<Arc> *ofst = std::get<0>(*args)->GetMutableFst<Arc>();
  const Fst<Arc> &ifst = *(std::get<1>(*args).GetFst<Arc>());
  Concat(ofst, ifst);
}

using ConcatArgs2 = std::pair<const FstClass &, MutableFstClass *>;

template <class Arc>
void Concat(ConcatArgs2 *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  Concat(ifst, ofst);
}

void Concat(MutableFstClass *ofst, const FstClass &ifst);

void Concat(const FstClass &ifst, MutableFstClass *ofst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CONCAT_H_
