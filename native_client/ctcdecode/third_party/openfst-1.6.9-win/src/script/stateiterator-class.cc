// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/script-impl.h>
#include <fst/script/stateiterator-class.h>

namespace fst {
namespace script {

StateIteratorClass::StateIteratorClass(const FstClass &fst) : impl_(nullptr) {
  InitStateIteratorClassArgs args(fst, this);
  Apply<Operation<InitStateIteratorClassArgs>>("InitStateIteratorClass",
                                               fst.ArcType(), &args);
}

REGISTER_FST_OPERATION(InitStateIteratorClass, StdArc,
                       InitStateIteratorClassArgs);
REGISTER_FST_OPERATION(InitStateIteratorClass, LogArc,
                       InitStateIteratorClassArgs);
REGISTER_FST_OPERATION(InitStateIteratorClass, Log64Arc,
                       InitStateIteratorClassArgs);

}  // namespace script
}  // namespace fst
