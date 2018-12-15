// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_TOPSORT_H_
#define FST_SCRIPT_TOPSORT_H_

#include <fst/topsort.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using TopSortArgs = WithReturnValue<bool, MutableFstClass *>;

template <class Arc>
void TopSort(TopSortArgs *args) {
  args->retval = TopSort(args->args->GetMutableFst<Arc>());
}

bool TopSort(MutableFstClass *fst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_TOPSORT_H_
