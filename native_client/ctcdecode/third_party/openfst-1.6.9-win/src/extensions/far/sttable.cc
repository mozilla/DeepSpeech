// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fstream>
#include <fst/extensions/far/sttable.h>

namespace fst {

bool IsSTTable(const string &filename) {
  std::ifstream strm(filename);
  if (!strm.good()) return false;

  int32 magic_number = 0;
  ReadType(strm, &magic_number);
  return magic_number == kSTTableMagicNumber;
}

}  // namespace fst
