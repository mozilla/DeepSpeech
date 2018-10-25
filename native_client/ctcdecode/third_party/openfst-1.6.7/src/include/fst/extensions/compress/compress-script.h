// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Declarations of 'scriptable' versions of compression operations, that is,
// those that can be called with FstClass-type arguments.

#ifndef FST_EXTENSIONS_COMPRESS_COMPRESS_SCRIPT_H_
#define FST_EXTENSIONS_COMPRESS_COMPRESS_SCRIPT_H_

#include <utility>
#include <vector>

#include <fst/extensions/compress/compress.h>

#include <fst/log.h>
#include <fst/util.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

typedef std::tuple<const FstClass &, const string &, const bool> CompressArgs;

template <class Arc>
void Compress(CompressArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  const string &filename = std::get<1>(*args);
  const bool gzip = std::get<2>(*args);

  if (!fst::Compress(fst, filename, gzip)) FSTERROR() << "Compress: failed";
}

void Compress(const FstClass &fst, const string &filename, const bool gzip);

typedef std::tuple<const string &, MutableFstClass *, const bool>
    DecompressArgs;

template <class Arc>
void Decompress(DecompressArgs *args) {
  const string &filename = std::get<0>(*args);
  MutableFst<Arc> *fst = std::get<1>(*args)->GetMutableFst<Arc>();
  const bool gzip = std::get<2>(*args);

  if (!fst::Decompress(filename, fst, gzip))
    FSTERROR() << "Decompress: failed";
}

void Decompress(const string &filename, MutableFstClass *fst, const bool gzip);

}  // namespace script
}  // namespace fst

#endif  // FST_EXTENSIONS_COMPRESS_COMPRESS_SCRIPT_H_
