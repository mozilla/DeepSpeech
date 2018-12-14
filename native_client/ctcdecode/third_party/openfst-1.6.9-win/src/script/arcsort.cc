// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/arcsort.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void ArcSort(MutableFstClass *fst, ArcSortType sort_type) {
  ArcSortArgs args(fst, sort_type);
  Apply<Operation<ArcSortArgs>>("ArcSort", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(ArcSort, StdArc, ArcSortArgs);
REGISTER_FST_OPERATION(ArcSort, LogArc, ArcSortArgs);
REGISTER_FST_OPERATION(ArcSort, Log64Arc, ArcSortArgs);

}  // namespace script
}  // namespace fst
