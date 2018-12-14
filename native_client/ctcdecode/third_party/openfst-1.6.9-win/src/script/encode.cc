// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/encode.h>
#include <fst/script/encode.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Encode(MutableFstClass *fst, uint32 flags, bool reuse_encoder,
            const string &coder_fname) {
  EncodeArgs1 args(fst, flags, reuse_encoder, coder_fname);
  Apply<Operation<EncodeArgs1>>("Encode", fst->ArcType(), &args);
}

void Encode(MutableFstClass *fst, EncodeMapperClass *encoder) {
  if (!internal::ArcTypesMatch(*fst, *encoder, "Encode")) {
    fst->SetProperties(kError, kError);
    return;
  }
  EncodeArgs2 args(fst, encoder);
  Apply<Operation<EncodeArgs2>>("Encode", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Encode, StdArc, EncodeArgs1);
REGISTER_FST_OPERATION(Encode, LogArc, EncodeArgs1);
REGISTER_FST_OPERATION(Encode, Log64Arc, EncodeArgs1);

REGISTER_FST_OPERATION(Encode, StdArc, EncodeArgs2);
REGISTER_FST_OPERATION(Encode, LogArc, EncodeArgs2);
REGISTER_FST_OPERATION(Encode, Log64Arc, EncodeArgs2);

}  // namespace script
}  // namespace fst
