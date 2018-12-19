// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CLOSURE_H_
#define FST_SCRIPT_CLOSURE_H_

#include <utility>

#include <fst/closure.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ClosureArgs = std::pair<MutableFstClass *, const ClosureType>;

template <class Arc>
void Closure(ClosureArgs *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  Closure(fst, std::get<1>(*args));
}

void Closure(MutableFstClass *ofst, ClosureType closure_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CLOSURE_H_
