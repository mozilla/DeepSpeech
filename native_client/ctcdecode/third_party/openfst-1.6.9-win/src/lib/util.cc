// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// FST utility definitions.

#include <fst/util.h>
#include <cctype>
#include <sstream>
#include <string>
#include <fst/flags.h>
#include <fst/log.h>
#include <fst/mapped-file.h>


// Utility flag definitions

DEFINE_bool(fst_error_fatal, true,
            "FST errors are fatal; o.w. return objects flagged as bad: "
            "e.g., FSTs: kError property set, FST weights: not a Member()");

namespace fst {

void SplitString(char *full, const char *delim, std::vector<char *> *vec,
                 bool omit_empty_strings) {
  char *p = full;
  while (p) {
    if ((p = strpbrk(full, delim))) {
      p[0] = '\0';
    }
    if (!omit_empty_strings || full[0] != '\0') vec->push_back(full);
    if (p) full = p + 1;
  }
}

int64_t StrToint64_t(const string &s, const string &src, size_t nline,
                 bool allow_negative, bool *error) {
  int64_t n;
  const char *cs = s.c_str();
  char *p;
  if (error) *error = false;
  n = strtoll(cs, &p, 10);
  if (p < cs + s.size() || (!allow_negative && n < 0)) {
    FSTERROR() << "StrToint64_t: Bad integer = " << s << "\", source = " << src
               << ", line = " << nline;
    if (error) *error = true;
    return 0;
  }
  return n;
}

void ConvertToLegalCSymbol(string *s) {
  for (auto it = s->begin(); it != s->end(); ++it) {
    if (!isalnum(*it)) {
      *it = '_';
    }
  }
}

// Skips over input characters to align to 'align' bytes. Returns false if can't
// align.
bool AlignInput(std::istream &strm) {
  char c;
  for (int i = 0; i < MappedFile::kArchAlignment; ++i) {
    int64_t pos = strm.tellg();
    if (pos < 0) {
      LOG(ERROR) << "AlignInput: Can't determine stream position";
      return false;
    }
    if (pos % MappedFile::kArchAlignment == 0) break;
    strm.read(&c, 1);
  }
  return true;
}

// Write null output characters to align to 'align' bytes. Returns false if
// can't align.
bool AlignOutput(std::ostream &strm) {
  for (int i = 0; i < MappedFile::kArchAlignment; ++i) {
    int64_t pos = strm.tellp();
    if (pos < 0) {
      LOG(ERROR) << "AlignOutput: Can't determine stream position";
      return false;
    }
    if (pos % MappedFile::kArchAlignment == 0) break;
    strm.write("", 1);
  }
  return true;
}

int AlignBufferWithOutputStream(std::ostream &strm,
                                std::ostringstream &buffer) {
  const auto strm_pos = strm.tellp();
  if (strm_pos == std::ostream::pos_type(-1)) {
    LOG(ERROR) << "Cannot determine stream position";
    return -1;
  }
  const int stream_offset = strm_pos % MappedFile::kArchAlignment;
  for (int i = 0; i < stream_offset; ++i) buffer.write("", 1);
  return stream_offset;
}

}  // namespace fst
