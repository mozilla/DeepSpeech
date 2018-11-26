// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/verify.h>

namespace fst {
namespace script {

bool Verify(const FstClass &fst) {
  VerifyArgs args(fst);
  Apply<Operation<VerifyArgs>>("Verify", fst.ArcType(), &args);
  return args.retval;
}

REGISTER_FST_OPERATION(Verify, StdArc, VerifyArgs);
REGISTER_FST_OPERATION(Verify, LogArc, VerifyArgs);
REGISTER_FST_OPERATION(Verify, Log64Arc, VerifyArgs);

}  // namespace script
}  // namespace fst
