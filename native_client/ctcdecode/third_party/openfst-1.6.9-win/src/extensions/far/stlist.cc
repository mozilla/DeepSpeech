// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <ios>

#include <fst/extensions/far/stlist.h>
#include <fstream>

namespace fst {

bool IsSTList(const string &filename) {
  std::ifstream strm(filename, std::ios_base::in | std::ios_base::binary);
  if (!strm) return false;
  int32 magic_number = 0;
  ReadType(strm, &magic_number);
  return magic_number == kSTListMagicNumber;
}

}  // namespace fst
