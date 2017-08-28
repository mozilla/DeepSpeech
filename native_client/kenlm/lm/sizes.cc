#include "lm/sizes.hh"
#include "lm/model.hh"
#include "util/file_piece.hh"

#include <vector>
#include <iomanip>

namespace lm {
namespace ngram {

void ShowSizes(const std::vector<uint64_t> &counts, const lm::ngram::Config &config) {
  uint64_t sizes[6];
  sizes[0] = ProbingModel::Size(counts, config);
  sizes[1] = RestProbingModel::Size(counts, config);
  sizes[2] = TrieModel::Size(counts, config);
  sizes[3] = QuantTrieModel::Size(counts, config);
  sizes[4] = ArrayTrieModel::Size(counts, config);
  sizes[5] = QuantArrayTrieModel::Size(counts, config);
  uint64_t max_length = *std::max_element(sizes, sizes + sizeof(sizes) / sizeof(uint64_t));
  uint64_t min_length = *std::min_element(sizes, sizes + sizeof(sizes) / sizeof(uint64_t));
  uint64_t divide;
  char prefix;
  if (min_length < (1 << 10) * 10) {
    prefix = ' ';
    divide = 1;
  } else if (min_length < (1 << 20) * 10) {
    prefix = 'k';
    divide = 1 << 10;
  } else if (min_length < (1ULL << 30) * 10) {
    prefix = 'M';
    divide = 1 << 20;
  } else {
    prefix = 'G';
    divide = 1 << 30;
  }
  long int length = std::max<long int>(2, static_cast<long int>(ceil(log10((double) max_length / divide))));
  std::cerr << "Memory estimate for binary LM:\ntype    ";

  // right align bytes.
  for (long int i = 0; i < length - 2; ++i) std::cerr << ' ';

  std::cerr << prefix << "B\n"
    "probing " << std::setw(length) << (sizes[0] / divide) << " assuming -p " << config.probing_multiplier << "\n"
    "probing " << std::setw(length) << (sizes[1] / divide) << " assuming -r models -p " << config.probing_multiplier << "\n"
    "trie    " << std::setw(length) << (sizes[2] / divide) << " without quantization\n"
    "trie    " << std::setw(length) << (sizes[3] / divide) << " assuming -q " << (unsigned)config.prob_bits << " -b " << (unsigned)config.backoff_bits << " quantization \n"
    "trie    " << std::setw(length) << (sizes[4] / divide) << " assuming -a " << (unsigned)config.pointer_bhiksha_bits << " array pointer compression\n"
    "trie    " << std::setw(length) << (sizes[5] / divide) << " assuming -a " << (unsigned)config.pointer_bhiksha_bits << " -q " << (unsigned)config.prob_bits << " -b " << (unsigned)config.backoff_bits<< " array pointer compression and quantization\n";
}

void ShowSizes(const std::vector<uint64_t> &counts) {
  lm::ngram::Config config;
  ShowSizes(counts, config);
}

void ShowSizes(const char *file, const lm::ngram::Config &config) {
  std::vector<uint64_t> counts;
  util::FilePiece f(file);
  lm::ReadARPACounts(f, counts);
  ShowSizes(counts, config);
}

}} //namespaces
