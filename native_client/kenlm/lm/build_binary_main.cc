#include "lm/model.hh"
#include "lm/sizes.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

#ifdef WIN32
#include "util/getopt.hh"
#else
#include <unistd.h>
#endif

namespace lm {
namespace ngram {
namespace {

void Usage(const char *name, const char *default_mem) {
  std::cerr << "Usage: " << name << " [-u log10_unknown_probability] [-s] [-i] [-v] [-w mmap|after] [-p probing_multiplier] [-T trie_temporary] [-S trie_building_mem] [-q bits] [-b bits] [-a bits] [type] input.arpa [output.mmap]\n\n"
"-u sets the log10 probability for <unk> if the ARPA file does not have one.\n"
"   Default is -100.  The ARPA file will always take precedence.\n"
"-s allows models to be built even if they do not have <s> and </s>.\n"
"-i allows buggy models from IRSTLM by mapping positive log probability to 0.\n"
"-v disables inclusion of the vocabulary in the binary file.\n"
"-w mmap|after determines how writing is done.\n"
"   mmap maps the binary file and writes to it.  Default for trie.\n"
"   after allocates anonymous memory, builds, and writes.  Default for probing.\n"
"-r \"order1.arpa order2 order3 order4\" adds lower-order rest costs from these\n"
"   model files.  order1.arpa must be an ARPA file.  All others may be ARPA or\n"
"   the same data structure as being built.  All files must have the same\n"
"   vocabulary.  For probing, the unigrams must be in the same order.\n\n"
"type is either probing or trie.  Default is probing.\n\n"
"probing uses a probing hash table.  It is the fastest but uses the most memory.\n"
"-p sets the space multiplier and must be >1.0.  The default is 1.5.\n\n"
"trie is a straightforward trie with bit-level packing.  It uses the least\n"
"memory and is still faster than SRI or IRST.  Building the trie format uses an\n"
"on-disk sort to save memory.\n"
"-T is the temporary directory prefix.  Default is the output file name.\n"
"-S determines memory use for sorting.  Default is " << default_mem << ".  This is compatible\n"
"   with GNU sort.  The number is followed by a unit: \% for percent of physical\n"
"   memory, b for bytes, K for Kilobytes, M for megabytes, then G,T,P,E,Z,Y.  \n"
"   Default unit is K for Kilobytes.\n"
"-q turns quantization on and sets the number of bits (e.g. -q 8).\n"
"-b sets backoff quantization bits.  Requires -q and defaults to that value.\n"
"-a compresses pointers using an array of offsets.  The parameter is the\n"
"   maximum number of bits encoded by the array.  Memory is minimized subject\n"
"   to the maximum, so pick 255 to minimize memory.\n\n"
"-h print this help message.\n\n"
"Get a memory estimate by passing an ARPA file without an output file name.\n";
  exit(1);
}

// I could really use boost::lexical_cast right about now.
float ParseFloat(const char *from) {
  char *end;
  float ret = strtod(from, &end);
  if (*end) throw util::ParseNumberException(from);
  return ret;
}
unsigned long int ParseUInt(const char *from) {
  char *end;
  unsigned long int ret = strtoul(from, &end, 10);
  if (*end) throw util::ParseNumberException(from);
  return ret;
}

uint8_t ParseBitCount(const char *from) {
  unsigned long val = ParseUInt(from);
  if (val > 25) {
    util::ParseNumberException e(from);
    e << " bit counts are limited to 25.";
  }
  return val;
}

void ParseFileList(const char *from, std::vector<std::string> &to) {
  to.clear();
  while (true) {
    const char *i;
    for (i = from; *i && *i != ' '; ++i) {}
    to.push_back(std::string(from, i - from));
    if (!*i) break;
    from = i + 1;
  }
}

void ProbingQuantizationUnsupported() {
  std::cerr << "Quantization is only implemented in the trie data structure." << std::endl;
  exit(1);
}

} // namespace ngram
} // namespace lm
} // namespace

int main(int argc, char *argv[]) {
  using namespace lm::ngram;

  const char *default_mem = util::GuessPhysicalMemory() ? "80%" : "1G";

  if (argc == 2 && !strcmp(argv[1], "--help"))
    Usage(argv[0], default_mem);

  try {
    bool quantize = false, set_backoff_bits = false, bhiksha = false, set_write_method = false, rest = false;
    lm::ngram::Config config;
    config.building_memory = util::ParseSize(default_mem);
    int opt;
    while ((opt = getopt(argc, argv, "q:b:a:u:p:t:T:m:S:w:sir:vh")) != -1) {
      switch(opt) {
        case 'q':
          config.prob_bits = ParseBitCount(optarg);
          if (!set_backoff_bits) config.backoff_bits = config.prob_bits;
          quantize = true;
          break;
        case 'b':
          config.backoff_bits = ParseBitCount(optarg);
          set_backoff_bits = true;
          break;
        case 'a':
          config.pointer_bhiksha_bits = ParseBitCount(optarg);
          bhiksha = true;
          break;
        case 'u':
          config.unknown_missing_logprob = ParseFloat(optarg);
          break;
        case 'p':
          config.probing_multiplier = ParseFloat(optarg);
          break;
        case 't': // legacy
        case 'T':
          config.temporary_directory_prefix = optarg;
          util::NormalizeTempPrefix(config.temporary_directory_prefix);
          break;
        case 'm': // legacy
          config.building_memory = ParseUInt(optarg) * 1048576;
          break;
        case 'S':
          config.building_memory = std::min(static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()), util::ParseSize(optarg));
          break;
        case 'w':
          set_write_method = true;
          if (!strcmp(optarg, "mmap")) {
            config.write_method = Config::WRITE_MMAP;
          } else if (!strcmp(optarg, "after")) {
            config.write_method = Config::WRITE_AFTER;
          } else {
            Usage(argv[0], default_mem);
          }
          break;
        case 's':
          config.sentence_marker_missing = lm::SILENT;
          break;
        case 'i':
          config.positive_log_probability = lm::SILENT;
          break;
        case 'r':
          rest = true;
          ParseFileList(optarg, config.rest_lower_files);
          config.rest_function = Config::REST_LOWER;
          break;
        case 'v':
          config.include_vocab = false;
          break;
        case 'h': // help
        default:
          Usage(argv[0], default_mem);
      }
    }
    if (!quantize && set_backoff_bits) {
      std::cerr << "You specified backoff quantization (-b) but not probability quantization (-q)" << std::endl;
      abort();
    }
    if (optind + 1 == argc) {
      ShowSizes(argv[optind], config);
      return 0;
    }
    const char *model_type;
    const char *from_file;

    if (optind + 2 == argc) {
      model_type = "probing";
      from_file = argv[optind];
      config.write_mmap = argv[optind + 1];
    } else if (optind + 3 == argc) {
      model_type = argv[optind];
      from_file = argv[optind + 1];
      config.write_mmap = argv[optind + 2];
    } else {
      Usage(argv[0], default_mem);
      return 1;
    }
    if (!strcmp(model_type, "probing")) {
      if (!set_write_method) config.write_method = Config::WRITE_AFTER;
      if (quantize || set_backoff_bits) ProbingQuantizationUnsupported();
      if (rest) {
        RestProbingModel(from_file, config);
      } else {
        ProbingModel(from_file, config);
      }
    } else if (!strcmp(model_type, "trie")) {
      if (rest) {
        std::cerr << "Rest + trie is not supported yet." << std::endl;
        return 1;
      }
      if (!set_write_method) config.write_method = Config::WRITE_MMAP;
      if (quantize) {
        if (bhiksha) {
          QuantArrayTrieModel(from_file, config);
        } else {
          QuantTrieModel(from_file, config);
        }
      } else {
        if (bhiksha) {
          ArrayTrieModel(from_file, config);
        } else {
          TrieModel(from_file, config);
        }
      }
    } else {
      Usage(argv[0], default_mem);
    }
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "ERROR" << std::endl;
    return 1;
  }
  std::cerr << "SUCCESS" << std::endl;
  return 0;
}
