// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/concat.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
void Concat(MutableFstClass *ofst, const FstClass &ifst) {
  if (!internal::ArcTypesMatch(*ofst, ifst, "Concat")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ConcatArgs1 args(ofst, ifst);
  Apply<Operation<ConcatArgs1>>("Concat", ofst->ArcType(), &args);
}

// 2
void Concat(const FstClass &ifst, MutableFstClass *ofst) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "Concat")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ConcatArgs2 args(ifst, ofst);
  Apply<Operation<ConcatArgs2>>("Concat", ofst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Concat, StdArc, ConcatArgs1);
REGISTER_FST_OPERATION(Concat, LogArc, ConcatArgs1);
REGISTER_FST_OPERATION(Concat, Log64Arc, ConcatArgs1);

REGISTER_FST_OPERATION(Concat, StdArc, ConcatArgs2);
REGISTER_FST_OPERATION(Concat, LogArc, ConcatArgs2);
REGISTER_FST_OPERATION(Concat, Log64Arc, ConcatArgs2);

}  // namespace script
}  // namespace fst
