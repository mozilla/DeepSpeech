// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CONNECT_H_
#define FST_SCRIPT_CONNECT_H_

#include <fst/connect.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

template <class Arc>
void Connect(MutableFstClass *fst) {
  Connect(fst->GetMutableFst<Arc>());
}

void Connect(MutableFstClass *fst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CONNECT_H_
