// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/shortest-path.h>

namespace fst {
namespace script {

void ShortestPath(const FstClass &ifst, MutableFstClass *ofst,
                  const ShortestPathOptions &opts) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "ShortestPath")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ShortestPathArgs args(ifst, ofst, opts);
  Apply<Operation<ShortestPathArgs>>("ShortestPath", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(ShortestPath, StdArc, ShortestPathArgs);
REGISTER_FST_OPERATION(ShortestPath, LogArc, ShortestPathArgs);
REGISTER_FST_OPERATION(ShortestPath, Log64Arc, ShortestPathArgs);

}  // namespace script
}  // namespace fst
