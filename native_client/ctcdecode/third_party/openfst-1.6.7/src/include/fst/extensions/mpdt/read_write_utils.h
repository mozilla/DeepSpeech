// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Definition of ReadLabelTriples based on ReadLabelPairs, like that in
// nlp/fst/lib/util.h for pairs, and similarly for WriteLabelTriples.

#ifndef FST_EXTENSIONS_MPDT_READ_WRITE_UTILS_H_
#define FST_EXTENSIONS_MPDT_READ_WRITE_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include <fstream>
#include <fst/test-properties.h>

namespace fst {

// Returns true on success.
template <typename Label>
bool ReadLabelTriples(const string &filename,
                      std::vector<std::pair<Label, Label>> *pairs,
                      std::vector<Label> *assignments,
                      bool allow_negative = false) {
  std::ifstream fstrm(filename);
  if (!fstrm) {
    LOG(ERROR) << "ReadIntTriples: Can't open file: " << filename;
    return false;
  }
  static constexpr auto kLineLen = 8096;
  char line[kLineLen];
  size_t nline = 0;
  pairs->clear();
  while (fstrm.getline(line, kLineLen)) {
    ++nline;
    std::vector<char *> col;
    SplitString(line, "\n\t ", &col, true);
    // Empty line or comment?
    if (col.empty() || col[0][0] == '\0' || col[0][0] == '#') continue;
    if (col.size() != 3) {
      LOG(ERROR) << "ReadLabelTriples: Bad number of columns, "
                 << "file = " << filename << ", line = " << nline;
      return false;
    }
    bool err;
    const Label i1 = StrToInt64(col[0], filename, nline, allow_negative, &err);
    if (err) return false;
    const Label i2 = StrToInt64(col[1], filename, nline, allow_negative, &err);
    if (err) return false;
    using Level = Label;
    const Level i3 = StrToInt64(col[2], filename, nline, allow_negative, &err);
    if (err) return false;
    pairs->push_back(std::make_pair(i1, i2));
    assignments->push_back(i3);
  }
  return true;
}

// Returns true on success.
template <typename Label>
bool WriteLabelTriples(const string &filename,
                       const std::vector<std::pair<Label, Label>> &pairs,
                       const std::vector<Label> &assignments) {
  if (pairs.size() != assignments.size()) {
    LOG(ERROR) << "WriteLabelTriples: Pairs and assignments of different sizes";
    return false;
  }
  std::ofstream fstrm(filename);
  if (!fstrm) {
    LOG(ERROR) << "WriteLabelTriples: Can't open file: " << filename;
    return false;
  }
  for (size_t n = 0; n < pairs.size(); ++n)
    fstrm << pairs[n].first << "\t" << pairs[n].second << "\t" << assignments[n]
          << "\n";
  if (!fstrm) {
    LOG(ERROR) << "WriteLabelTriples: Write failed: "
               << (filename.empty() ? "standard output" : filename);
    return false;
  }
  return true;
}

}  // namespace fst

#endif  // FST_EXTENSIONS_MPDT_READ_WRITE_UTILS_H_
