// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/topsort.h>

namespace fst {
namespace script {

bool TopSort(MutableFstClass *fst) {
  TopSortArgs args(fst);
  Apply<Operation<TopSortArgs>>("TopSort", fst->ArcType(), &args);
  return args.retval;
}

REGISTER_FST_OPERATION(TopSort, StdArc, TopSortArgs);
REGISTER_FST_OPERATION(TopSort, LogArc, TopSortArgs);
REGISTER_FST_OPERATION(TopSort, Log64Arc, TopSortArgs);

}  // namespace script
}  // namespace fst
