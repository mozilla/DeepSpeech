// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/minimize.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Minimize(MutableFstClass *ofst1, MutableFstClass *ofst2, float delta,
              bool allow_nondet) {
  if (ofst2 && !internal::ArcTypesMatch(*ofst1, *ofst2, "Minimize")) {
    ofst1->SetProperties(kError, kError);
    ofst2->SetProperties(kError, kError);
    return;
  }
  MinimizeArgs args(ofst1, ofst2, delta, allow_nondet);
  Apply<Operation<MinimizeArgs>>("Minimize", ofst1->ArcType(), &args);
}

REGISTER_FST_OPERATION(Minimize, StdArc, MinimizeArgs);
REGISTER_FST_OPERATION(Minimize, LogArc, MinimizeArgs);
REGISTER_FST_OPERATION(Minimize, Log64Arc, MinimizeArgs);

}  // namespace script
}  // namespace fst
