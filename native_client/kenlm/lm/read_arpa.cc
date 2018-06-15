#include "lm/read_arpa.hh"

#include "lm/blank.hh"
#include "util/file.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include <cctype>
#include <cstring>
#include <stdint.h>

#ifdef WIN32
#include <float.h>
#endif

namespace lm {

// 1 for '\t', '\n', and ' '.  This is stricter than isspace.
const bool kARPASpaces[256] = {0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

namespace {

bool IsEntirelyWhiteSpace(const StringPiece &line) {
  for (size_t i = 0; i < static_cast<size_t>(line.size()); ++i) {
    if (!isspace(line.data()[i])) return false;
  }
  return true;
}

const char kBinaryMagic[] = "mmap lm http://kheafield.com/code";

// strtoull isn't portable enough :-(
uint64_t ReadCount(const std::string &from) {
  std::stringstream stream(from);
  uint64_t ret;
  stream >> ret;
  UTIL_THROW_IF(!stream, FormatLoadException, "Bad count " << from);
  return ret;
}

} // namespace

void ReadARPACounts(util::FilePiece &in, std::vector<uint64_t> &number) {
  number.clear();
  StringPiece line = in.ReadLine();
  // In general, ARPA files can have arbitrary text before "\data\"
  // But in KenLM, we require such lines to start with "#", so that
  // we can do stricter error checking
  while (IsEntirelyWhiteSpace(line) || starts_with(line, "#")) {
    line = in.ReadLine();
  }

  if (line != "\\data\\") {
    if ((line.size() >= 2) && (line.data()[0] == 0x1f) && (static_cast<unsigned char>(line.data()[1]) == 0x8b)) {
      UTIL_THROW(FormatLoadException, "Looks like a gzip file.  If this is an ARPA file, pipe " << in.FileName() << " through zcat.  If this already in binary format, you need to decompress it because mmap doesn't work on top of gzip.");
    }
    if (static_cast<size_t>(line.size()) >= strlen(kBinaryMagic) && StringPiece(line.data(), strlen(kBinaryMagic)) == kBinaryMagic)
      UTIL_THROW(FormatLoadException, "This looks like a binary file but got sent to the ARPA parser.  Did you compress the binary file or pass a binary file where only ARPA files are accepted?");
    UTIL_THROW_IF(line.size() >= 4 && StringPiece(line.data(), 4) == "blmt", FormatLoadException, "This looks like an IRSTLM binary file.  Did you forget to pass --text yes to compile-lm?");
    UTIL_THROW_IF(line == "iARPA", FormatLoadException, "This looks like an IRSTLM iARPA file.  You need an ARPA file.  Run\n  compile-lm --text yes " << in.FileName() << " " << in.FileName() << ".arpa\nfirst.");
    UTIL_THROW(FormatLoadException, "first non-empty line was \"" << line << "\" not \\data\\.");
  }
  while (!IsEntirelyWhiteSpace(line = in.ReadLine())) {
    if (line.size() < 6 || strncmp(line.data(), "ngram ", 6)) UTIL_THROW(FormatLoadException, "count line \"" << line << "\"doesn't begin with \"ngram \"");
    // So strtol doesn't go off the end of line.
    std::string remaining(line.data() + 6, line.size() - 6);
    char *end_ptr;
    unsigned int length = std::strtol(remaining.c_str(), &end_ptr, 10);
    if ((end_ptr == remaining.c_str()) || (length - 1 != number.size())) UTIL_THROW(FormatLoadException, "ngram count lengths should be consecutive starting with 1: " << line);
    if (*end_ptr != '=') UTIL_THROW(FormatLoadException, "Expected = immediately following the first number in the count line " << line);
    ++end_ptr;
    number.push_back(ReadCount(end_ptr));
  }
}

void ReadNGramHeader(util::FilePiece &in, unsigned int length) {
   StringPiece line;
  while (IsEntirelyWhiteSpace(line = in.ReadLine())) {}
  std::stringstream expected;
  expected << '\\' << length << "-grams:";
  if (line != expected.str()) UTIL_THROW(FormatLoadException, "Was expecting n-gram header " << expected.str() << " but got " << line << " instead");
}

void ReadBackoff(util::FilePiece &in, Prob &/*weights*/) {
  switch (in.get()) {
    case '\t':
      {
        float got = in.ReadFloat();
        if (got != 0.0)
          UTIL_THROW(FormatLoadException, "Non-zero backoff " << got << " provided for an n-gram that should have no backoff");
      }
      break;
    case '\n':
      break;
    default:
      UTIL_THROW(FormatLoadException, "Expected tab or newline for backoff");
  }
}

void ReadBackoff(util::FilePiece &in, float &backoff) {
  // Always make zero negative.
  // Negative zero means that no (n+1)-gram has this n-gram as context.
  // Therefore the hypothesis state can be shorter.  Of course, many n-grams
  // are context for (n+1)-grams.  An algorithm in the data structure will go
  // back and set the backoff to positive zero in these cases.
  switch (in.get()) {
    case '\t':
      backoff = in.ReadFloat();
      if (backoff == ngram::kExtensionBackoff) backoff = ngram::kNoExtensionBackoff;
      {
#if defined(WIN32) && !defined(__MINGW32__)
		int float_class = _fpclass(backoff);
        UTIL_THROW_IF(float_class == _FPCLASS_SNAN || float_class == _FPCLASS_QNAN || float_class == _FPCLASS_NINF || float_class == _FPCLASS_PINF, FormatLoadException, "Bad backoff " << backoff);
#else
        int float_class = std::fpclassify(backoff);
        UTIL_THROW_IF(float_class == FP_NAN || float_class == FP_INFINITE, FormatLoadException, "Bad backoff " << backoff);
#endif
      }
      UTIL_THROW_IF(in.get() != '\n', FormatLoadException, "Expected newline after backoff");
      break;
    case '\n':
      backoff = ngram::kNoExtensionBackoff;
      break;
    default:
      UTIL_THROW(FormatLoadException, "Expected tab or newline for backoff");
  }
}

void ReadEnd(util::FilePiece &in) {
  StringPiece line;
  do {
    line = in.ReadLine();
  } while (IsEntirelyWhiteSpace(line));
  if (line != "\\end\\") UTIL_THROW(FormatLoadException, "Expected \\end\\ but the ARPA file has " << line);

  try {
    while (true) {
      line = in.ReadLine();
      if (!IsEntirelyWhiteSpace(line)) UTIL_THROW(FormatLoadException, "Trailing line " << line);
    }
  } catch (const util::EndOfFileException &e) {}
}

void PositiveProbWarn::Warn(float prob) {
  switch (action_) {
    case THROW_UP:
      UTIL_THROW(FormatLoadException, "Positive log probability " << prob << " in the model.  This is a bug in IRSTLM; you can set config.positive_log_probability = SILENT or pass -i to build_binary to substitute 0.0 for the log probability.  Error");
    case COMPLAIN:
      std::cerr << "There's a positive log probability " << prob << " in the APRA file, probably because of a bug in IRSTLM.  This and subsequent entires will be mapped to 0 log probability." << std::endl;
      action_ = SILENT;
      break;
    case SILENT:
      break;
  }
}

} // namespace lm
