// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/text-io.h>

#include <cstring>
#include <fstream>
#include <ostream>
#include <sstream>
#include <utility>

#include <fst/log.h>
#include <fstream>
#include <fst/util.h>

namespace fst {
namespace script {

// Reads vector of weights; returns true on success.
bool ReadPotentials(const string &weight_type, const string &filename,
                    std::vector<WeightClass> *potentials) {
  std::ifstream istrm(filename);
  if (!istrm.good()) {
    LOG(ERROR) << "ReadPotentials: Can't open file: " << filename;
    return false;
  }
  static constexpr int kLineLen = 8096;
  char line[kLineLen];
  size_t nline = 0;
  potentials->clear();
  while (!istrm.getline(line, kLineLen).fail()) {
    ++nline;
    std::vector<char *> col;
    SplitString(line, "\n\t ", &col, true);
    if (col.empty() || col[0][0] == '\0') continue;
    if (col.size() != 2) {
      FSTERROR() << "ReadPotentials: Bad number of columns, "
                 << "file = " << filename << ", line = " << nline;
      return false;
    }
    const ssize_t s = StrToInt64(col[0], filename, nline, false);
    const WeightClass weight(weight_type, col[1]);
    while (potentials->size() <= s) {
      potentials->push_back(WeightClass::Zero(weight_type));
    }
    potentials->back() = weight;
  }
  return true;
}

// Writes vector of weights; returns true on success.
bool WritePotentials(const string &filename,
                     const std::vector<WeightClass> &potentials) {
  std::ofstream ostrm;
  if (!filename.empty()) {
    ostrm.open(filename);
    if (!ostrm.good()) {
      LOG(ERROR) << "WritePotentials: Can't open file: " << filename;
      return false;
    }
  }
  std::ostream &strm = ostrm.is_open() ? ostrm : std::cout;
  strm.precision(9);
  for (size_t s = 0; s < potentials.size(); ++s) {
    strm << s << "\t" << potentials[s] << "\n";
  }
  if (strm.fail()) {
    LOG(ERROR) << "WritePotentials: Write failed: "
               << (filename.empty() ? "standard output" : filename);
    return false;
  }
  return true;
}

}  // namespace script
}  // namespace fst
