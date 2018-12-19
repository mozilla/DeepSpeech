// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/compose.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Compose(const FstClass &ifst1, const FstClass &ifst2,
             MutableFstClass *ofst, const ComposeOptions &opts) {
  if (!internal::ArcTypesMatch(ifst1, ifst2, "Compose") ||
      !internal::ArcTypesMatch(*ofst, ifst1, "Compose")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ComposeArgs args(ifst1, ifst2, ofst, opts);
  Apply<Operation<ComposeArgs>>("Compose", ifst1.ArcType(), &args);
}

REGISTER_FST_OPERATION(Compose, StdArc, ComposeArgs);
REGISTER_FST_OPERATION(Compose, LogArc, ComposeArgs);
REGISTER_FST_OPERATION(Compose, Log64Arc, ComposeArgs);

}  // namespace script
}  // namespace fst
