// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/encode.h>
#include <fst/script/decode.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Decode(MutableFstClass *fst, const string &coder_fname) {
  DecodeArgs1 args(fst, coder_fname);
  Apply<Operation<DecodeArgs1>>("Decode", fst->ArcType(), &args);
}

void Decode(MutableFstClass *fst, const EncodeMapperClass &encoder) {
  if (!internal::ArcTypesMatch(*fst, encoder, "Decode")) {
    fst->SetProperties(kError, kError);
    return;
  }
  DecodeArgs2 args(fst, encoder);
  Apply<Operation<DecodeArgs2>>("Decode", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Decode, StdArc, DecodeArgs1);
REGISTER_FST_OPERATION(Decode, LogArc, DecodeArgs1);
REGISTER_FST_OPERATION(Decode, Log64Arc, DecodeArgs1);

REGISTER_FST_OPERATION(Decode, StdArc, DecodeArgs2);
REGISTER_FST_OPERATION(Decode, LogArc, DecodeArgs2);
REGISTER_FST_OPERATION(Decode, Log64Arc, DecodeArgs2);

}  // namespace script
}  // namespace fst
