// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/invert.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Invert(MutableFstClass *fst) {
  Apply<Operation<MutableFstClass>>("Invert", fst->ArcType(), fst);
}

REGISTER_FST_OPERATION(Invert, StdArc, MutableFstClass);
REGISTER_FST_OPERATION(Invert, LogArc, MutableFstClass);
REGISTER_FST_OPERATION(Invert, Log64Arc, MutableFstClass);

}  // namespace script
}  // namespace fst
