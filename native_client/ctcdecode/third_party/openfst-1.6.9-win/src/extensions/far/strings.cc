// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <cmath>
#include <string>

#include <fst/flags.h>
#include <fst/extensions/far/compile-strings.h>
#include <fstream>

DEFINE_string(far_field_separator, "\t",
              "Set of characters used as a separator between printed fields");

namespace fst {

// Computes the minimal length required to encode each line number as a decimal
// number.
int KeySize(const char *filename) {
  std::ifstream istrm(filename);
  istrm.seekg(0);
  string s;
  int nline = 0;
  while (getline(istrm, s)) ++nline;
  istrm.seekg(0);
  return nline ? ceil(log10(nline + 1)) : 1;
}

}  // namespace fst
