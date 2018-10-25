// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/intersect.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Intersect(const FstClass &ifst1, const FstClass &ifst2,
               MutableFstClass *ofst, const ComposeOptions &opts) {
  if (!internal::ArcTypesMatch(ifst1, ifst2, "Intersect") ||
      !internal::ArcTypesMatch(*ofst, ifst1, "Intersect")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  IntersectArgs args(ifst1, ifst2, ofst, opts);
  Apply<Operation<IntersectArgs>>("Intersect", ifst1.ArcType(), &args);
}

REGISTER_FST_OPERATION(Intersect, StdArc, IntersectArgs);
REGISTER_FST_OPERATION(Intersect, LogArc, IntersectArgs);
REGISTER_FST_OPERATION(Intersect, Log64Arc, IntersectArgs);

}  // namespace script
}  // namespace fst
