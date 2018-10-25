// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_INVERT_H_
#define FST_SCRIPT_INVERT_H_

#include <fst/invert.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

template <class Arc>
void Invert(MutableFstClass *fst) {
  Invert(fst->GetMutableFst<Arc>());
}

void Invert(MutableFstClass *fst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_INVERT_H_
