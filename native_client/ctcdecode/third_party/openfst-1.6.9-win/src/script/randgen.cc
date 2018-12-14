// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/randgen.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void RandGen(const FstClass &ifst, MutableFstClass *ofst, time_t seed,
             const RandGenOptions<RandArcSelection> &opts) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "RandGen")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  RandGenArgs args(ifst, ofst, seed, opts);
  Apply<Operation<RandGenArgs>>("RandGen", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(RandGen, StdArc, RandGenArgs);
REGISTER_FST_OPERATION(RandGen, LogArc, RandGenArgs);
REGISTER_FST_OPERATION(RandGen, Log64Arc, RandGenArgs);

}  // namespace script
}  // namespace fst
