// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_SYNCHRONIZE_H_
#define FST_SCRIPT_SYNCHRONIZE_H_

#include <utility>

#include <fst/synchronize.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using SynchronizeArgs = std::pair<const FstClass &, MutableFstClass *>;

template <class Arc>
void Synchronize(SynchronizeArgs *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  Synchronize(ifst, ofst);
}

void Synchronize(const FstClass &ifst, MutableFstClass *ofst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_SYNCHRONIZE_H_
