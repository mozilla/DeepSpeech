// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/determinize.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Determinize(const FstClass &ifst, MutableFstClass *ofst,
                 const DeterminizeOptions &opts) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "Determinize") ||
      !ofst->WeightTypesMatch(opts.weight_threshold, "Determinize")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DeterminizeArgs args(ifst, ofst, opts);
  Apply<Operation<DeterminizeArgs>>("Determinize", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(Determinize, StdArc, DeterminizeArgs);
REGISTER_FST_OPERATION(Determinize, LogArc, DeterminizeArgs);
REGISTER_FST_OPERATION(Determinize, Log64Arc, DeterminizeArgs);

}  // namespace script
}  // namespace fst
