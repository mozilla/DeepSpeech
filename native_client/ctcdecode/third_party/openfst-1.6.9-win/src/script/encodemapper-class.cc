// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/encodemapper-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

EncodeMapperClass::EncodeMapperClass(const string &arc_type, uint32 flags,
                                      EncodeType type) : impl_(nullptr) {
  InitEncodeMapperClassArgs args(flags, type, this);
  Apply<Operation<InitEncodeMapperClassArgs>>("InitEncodeMapperClass",
                                              arc_type, &args);
}

REGISTER_FST_OPERATION(InitEncodeMapperClass, StdArc,
                       InitEncodeMapperClassArgs);
REGISTER_FST_OPERATION(InitEncodeMapperClass, LogArc,
                       InitEncodeMapperClassArgs);
REGISTER_FST_OPERATION(InitEncodeMapperClass, Log64Arc,
                       InitEncodeMapperClassArgs);

}  // namespace script
}  // namespace fst
