// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/equal.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

bool Equal(const FstClass &fst1, const FstClass &fst2, float delta) {
  if (!internal::ArcTypesMatch(fst1, fst2, "Equal")) return false;
  EqualInnerArgs iargs(fst1, fst2, delta);
  EqualArgs args(iargs);
  Apply<Operation<EqualArgs>>("Equal", fst1.ArcType(), &args);
  return args.retval;
}

REGISTER_FST_OPERATION(Equal, StdArc, EqualArgs);
REGISTER_FST_OPERATION(Equal, LogArc, EqualArgs);
REGISTER_FST_OPERATION(Equal, Log64Arc, EqualArgs);

}  // namespace script
}  // namespace fst
