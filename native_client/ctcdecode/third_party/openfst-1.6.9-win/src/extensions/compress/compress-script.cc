// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Definitions of 'scriptable' versions of compression operations, that is,
// those that can be called with FstClass-type arguments.
//
// See comments in nlp/fst/script/script-impl.h for how the registration
// mechanism allows these to work with various arc types.

#include <fst/extensions/compress/compress-script.h>

#include <fst/arc-map.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Compress(const FstClass &fst, const string &filename, const bool gzip) {
  CompressArgs args(fst, filename, gzip);
  Apply<Operation<CompressArgs>>("Compress", fst.ArcType(), &args);
}

void Decompress(const string &filename, MutableFstClass *fst, const bool gzip) {
  DecompressArgs args(filename, fst, gzip);
  Apply<Operation<DecompressArgs>>("Decompress", fst->ArcType(), &args);
}

// Register operations for common arc types.

REGISTER_FST_OPERATION(Compress, StdArc, CompressArgs);
REGISTER_FST_OPERATION(Compress, LogArc, CompressArgs);
REGISTER_FST_OPERATION(Compress, Log64Arc, CompressArgs);

REGISTER_FST_OPERATION(Decompress, StdArc, DecompressArgs);
REGISTER_FST_OPERATION(Decompress, LogArc, DecompressArgs);
REGISTER_FST_OPERATION(Decompress, Log64Arc, DecompressArgs);

}  // namespace script
}  // namespace fst
