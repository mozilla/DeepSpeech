// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ENCODE_H_
#define FST_SCRIPT_ENCODE_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <fst/encode.h>
#include <fst/script/encodemapper-class.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using EncodeArgs1 = std::tuple<MutableFstClass *, uint32_t, bool, const string &>;

template <class Arc>
void Encode(EncodeArgs1 *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  const string &coder_fname = std::get<3>(*args);
  // If true, reuse encode from disk. If false, make a new encoder and just use
  // the filename argument as the destination state.
  std::unique_ptr<EncodeMapper<Arc>> encoder(
      std::get<2>(*args) ? EncodeMapper<Arc>::Read(coder_fname, ENCODE)
                         : new EncodeMapper<Arc>(std::get<1>(*args), ENCODE));
  Encode(fst, encoder.get());
  if (!std::get<2>(*args)) encoder->Write(coder_fname);
}

using EncodeArgs2 = std::pair<MutableFstClass *, EncodeMapperClass *>;

template <class Arc>
void Encode(EncodeArgs2 *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  EncodeMapper<Arc> *encoder = std::get<1>(*args)->GetEncodeMapper<Arc>();
  Encode(fst, encoder);
}

void Encode(MutableFstClass *fst, uint32_t flags, bool reuse_encoder,
            const string &coder_fname);

void Encode(MutableFstClass *fst, EncodeMapperClass *encoder);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ENCODE_H_
