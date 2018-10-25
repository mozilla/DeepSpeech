// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/epsnormalize.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void EpsNormalize(const FstClass &ifst, MutableFstClass *ofst,
                  EpsNormalizeType norm_type) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "EpsNormalize")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  EpsNormalizeArgs args(ifst, ofst, norm_type);
  Apply<Operation<EpsNormalizeArgs>>("EpsNormalize", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(EpsNormalize, StdArc, EpsNormalizeArgs);
REGISTER_FST_OPERATION(EpsNormalize, LogArc, EpsNormalizeArgs);
REGISTER_FST_OPERATION(EpsNormalize, Log64Arc, EpsNormalizeArgs);

}  // namespace script
}  // namespace fst
