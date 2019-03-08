// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/connect.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Connect(MutableFstClass *fst) {
  Apply<Operation<MutableFstClass>>("Connect", fst->ArcType(), fst);
}

REGISTER_FST_OPERATION(Connect, StdArc, MutableFstClass);
REGISTER_FST_OPERATION(Connect, LogArc, MutableFstClass);
REGISTER_FST_OPERATION(Connect, Log64Arc, MutableFstClass);

}  // namespace script
}  // namespace fst
