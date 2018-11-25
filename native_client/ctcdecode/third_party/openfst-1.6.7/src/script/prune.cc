// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/prune.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Prune(const FstClass &ifst, MutableFstClass *ofst,
           const WeightClass &weight_threshold,
           int64 state_threshold, float delta) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "Prune") ||
      !ofst->WeightTypesMatch(weight_threshold, "Prune")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  PruneArgs1 args(ifst, ofst, weight_threshold, state_threshold, delta);
  Apply<Operation<PruneArgs1>>("Prune", ifst.ArcType(), &args);
}

void Prune(MutableFstClass *fst, const WeightClass &weight_threshold,
           int64 state_threshold, float delta) {
  if (!fst->WeightTypesMatch(weight_threshold, "Prune")) {
    fst->SetProperties(kError, kError);
    return;
  }
  PruneArgs2 args(fst, weight_threshold, state_threshold, delta);
  Apply<Operation<PruneArgs2>>("Prune", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Prune, StdArc, PruneArgs1);
REGISTER_FST_OPERATION(Prune, LogArc, PruneArgs1);
REGISTER_FST_OPERATION(Prune, Log64Arc, PruneArgs1);

REGISTER_FST_OPERATION(Prune, StdArc, PruneArgs2);
REGISTER_FST_OPERATION(Prune, LogArc, PruneArgs2);
REGISTER_FST_OPERATION(Prune, Log64Arc, PruneArgs2);

}  // namespace script
}  // namespace fst
