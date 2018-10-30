// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_REVERSE_H_
#define FST_SCRIPT_REVERSE_H_

#include <tuple>

#include <fst/reverse.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ReverseArgs = std::tuple<const FstClass &, MutableFstClass *, bool>;

template <class Arc>
void Reverse(ReverseArgs *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  Reverse(ifst, ofst, std::get<2>(*args));
}

void Reverse(const FstClass &ifst, MutableFstClass *ofst,
             bool require_superinitial = true);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_REVERSE_H_
