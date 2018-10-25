// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DECODE_H_
#define FST_SCRIPT_DECODE_H_

#include <memory>
#include <string>
#include <utility>

#include <fst/encode.h>
#include <fst/script/encodemapper-class.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using DecodeArgs1 = std::pair<MutableFstClass *, const string &>;

template <class Arc>
void Decode(DecodeArgs1 *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  std::unique_ptr<EncodeMapper<Arc>> decoder(
      EncodeMapper<Arc>::Read(std::get<1>(*args), DECODE));
  if (!decoder) {
    fst->SetProperties(kError, kError);
    return;
  }
  Decode(fst, *decoder);
}

using DecodeArgs2 = std::pair<MutableFstClass *, const EncodeMapperClass &>;

template <class Arc>
void Decode(DecodeArgs2 *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  const EncodeMapper<Arc> &encoder =
      *(std::get<1>(*args).GetEncodeMapper<Arc>());
  Decode(fst, encoder);
}

void Decode(MutableFstClass *fst, const string &coder_fname);

void Decode(MutableFstClass *fst, const EncodeMapperClass &encoder);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DECODE_H_
