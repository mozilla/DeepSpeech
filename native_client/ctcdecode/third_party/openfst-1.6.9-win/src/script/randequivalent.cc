// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/randequivalent.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

bool RandEquivalent(const FstClass &fst1, const FstClass &fst2, int32 npath,
                    float delta, time_t seed,
                    const RandGenOptions<RandArcSelection> &opts) {
  if (!internal::ArcTypesMatch(fst1, fst2, "RandEquivalent")) return false;
  RandEquivalentInnerArgs iargs(fst1, fst2, npath, delta, seed, opts);
  RandEquivalentArgs args(iargs);
  Apply<Operation<RandEquivalentArgs>>("RandEquivalent", fst1.ArcType(), &args);
  return args.retval;
}

REGISTER_FST_OPERATION(RandEquivalent, StdArc, RandEquivalentArgs);
REGISTER_FST_OPERATION(RandEquivalent, LogArc, RandEquivalentArgs);
REGISTER_FST_OPERATION(RandEquivalent, Log64Arc, RandEquivalentArgs);

}  // namespace script
}  // namespace fst
