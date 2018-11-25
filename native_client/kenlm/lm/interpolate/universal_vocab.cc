#include "lm/interpolate/universal_vocab.hh"

namespace lm {
namespace interpolate {

UniversalVocab::UniversalVocab(const std::vector<WordIndex>& model_vocab_sizes) {
  model_index_map_.resize(model_vocab_sizes.size());
  for (size_t i = 0; i < model_vocab_sizes.size(); ++i) {
    model_index_map_[i].resize(model_vocab_sizes[i]);
  }
}

}} // namespaces
