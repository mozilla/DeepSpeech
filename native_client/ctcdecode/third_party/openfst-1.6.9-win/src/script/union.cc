// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/union.h>

namespace fst {
namespace script {

void Union(MutableFstClass *fst1, const FstClass &fst2) {
  if (!internal::ArcTypesMatch(*fst1, fst2, "Union")) {
    fst1->SetProperties(kError, kError);
    return;
  }
  UnionArgs args(fst1, fst2);
  Apply<Operation<UnionArgs>>("Union", fst1->ArcType(), &args);
}

REGISTER_FST_OPERATION(Union, StdArc, UnionArgs);
REGISTER_FST_OPERATION(Union, LogArc, UnionArgs);
REGISTER_FST_OPERATION(Union, Log64Arc, UnionArgs);

}  // namespace script
}  // namespace fst
