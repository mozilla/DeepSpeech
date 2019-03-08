// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/reweight.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Reweight(MutableFstClass *fst, const std::vector<WeightClass> &potential,
              ReweightType reweight_type) {
  ReweightArgs args(fst, potential, reweight_type);
  Apply<Operation<ReweightArgs>>("Reweight", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Reweight, StdArc, ReweightArgs);
REGISTER_FST_OPERATION(Reweight, LogArc, ReweightArgs);
REGISTER_FST_OPERATION(Reweight, Log64Arc, ReweightArgs);

}  // namespace script
}  // namespace fst
