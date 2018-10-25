// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/equivalent.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

bool Equivalent(const FstClass &fst1, const FstClass &fst2, float delta) {
  if (!internal::ArcTypesMatch(fst1, fst2, "Equivalent")) return false;
  EquivalentInnerArgs iargs(fst1, fst2, delta);
  EquivalentArgs args(iargs);
  Apply<Operation<EquivalentArgs>>("Equivalent", fst1.ArcType(), &args);
  return args.retval;
}

REGISTER_FST_OPERATION(Equivalent, StdArc, EquivalentArgs);
REGISTER_FST_OPERATION(Equivalent, LogArc, EquivalentArgs);
REGISTER_FST_OPERATION(Equivalent, Log64Arc, EquivalentArgs);

}  // namespace script
}  // namespace fst
