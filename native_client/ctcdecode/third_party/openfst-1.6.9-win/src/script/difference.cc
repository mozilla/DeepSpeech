// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/difference.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Difference(const FstClass &ifst1, const FstClass &ifst2,
                MutableFstClass *ofst, const ComposeOptions &opts) {
  if (!internal::ArcTypesMatch(ifst1, ifst2, "Difference") ||
      !internal::ArcTypesMatch(*ofst, ifst1, "Difference")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DifferenceArgs args(ifst1, ifst2, ofst, opts);
  Apply<Operation<DifferenceArgs>>("Difference", ifst1.ArcType(), &args);
}

REGISTER_FST_OPERATION(Difference, StdArc, DifferenceArgs);
REGISTER_FST_OPERATION(Difference, LogArc, DifferenceArgs);
REGISTER_FST_OPERATION(Difference, Log64Arc, DifferenceArgs);

}  // namespace script
}  // namespace fst
