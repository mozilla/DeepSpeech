// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/arciterator-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

ArcIteratorClass::ArcIteratorClass(const FstClass &fst, int64 s)
    : impl_(nullptr) {
  InitArcIteratorClassArgs args(fst, s, this);
  Apply<Operation<InitArcIteratorClassArgs>>("InitArcIteratorClass",
                                             fst.ArcType(), &args);
}

MutableArcIteratorClass::MutableArcIteratorClass(MutableFstClass *fst,
                                                 int64 s) : impl_(nullptr) {
  InitMutableArcIteratorClassArgs args(fst, s, this);
  Apply<Operation<InitMutableArcIteratorClassArgs>>(
      "InitMutableArcIteratorClass", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(InitArcIteratorClass, StdArc, InitArcIteratorClassArgs);
REGISTER_FST_OPERATION(InitArcIteratorClass, LogArc, InitArcIteratorClassArgs);
REGISTER_FST_OPERATION(InitArcIteratorClass, Log64Arc,
                       InitArcIteratorClassArgs);

REGISTER_FST_OPERATION(InitMutableArcIteratorClass, StdArc,
                       InitMutableArcIteratorClassArgs);
REGISTER_FST_OPERATION(InitMutableArcIteratorClass, LogArc,
                       InitMutableArcIteratorClassArgs);
REGISTER_FST_OPERATION(InitMutableArcIteratorClass, Log64Arc,
                       InitMutableArcIteratorClassArgs);

}  // namespace script
}  // namespace fst
