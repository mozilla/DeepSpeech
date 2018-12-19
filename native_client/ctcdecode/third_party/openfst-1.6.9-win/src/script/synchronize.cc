// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/synchronize.h>

namespace fst {
namespace script {

void Synchronize(const FstClass &ifst, MutableFstClass *ofst) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "Synchronize")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  SynchronizeArgs args(ifst, ofst);
  Apply<Operation<SynchronizeArgs>>("Synchronize", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(Synchronize, StdArc, SynchronizeArgs);
REGISTER_FST_OPERATION(Synchronize, LogArc, SynchronizeArgs);
REGISTER_FST_OPERATION(Synchronize, Log64Arc, SynchronizeArgs);

}  // namespace script
}  // namespace fst
