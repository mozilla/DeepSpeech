#ifndef LM_INTERPOLATE_UNIVERSAL_VOCAB_H
#define LM_INTERPOLATE_UNIVERSAL_VOCAB_H

#include "lm/word_index.hh"

#include <vector>
#include <cstddef>

namespace lm {
namespace interpolate {

class UniversalVocab {
public:
  explicit UniversalVocab(const std::vector<WordIndex>& model_vocab_sizes);

  // GetUniversalIndex takes the model number and index for the specific
  // model and returns the universal model number
  WordIndex GetUniversalIdx(std::size_t model_num, WordIndex model_word_index) const {
    return model_index_map_[model_num][model_word_index];
  }

  const WordIndex *Mapping(std::size_t model) const {
    return &*model_index_map_[model].begin();
  }

  WordIndex SlowConvertToModel(std::size_t model, WordIndex index) const {
    std::vector<WordIndex>::const_iterator i = lower_bound(model_index_map_[model].begin(), model_index_map_[model].end(), index);
    if (i == model_index_map_[model].end() || *i != index) return 0;
    return i - model_index_map_[model].begin();
  }

  void InsertUniversalIdx(std::size_t model_num, WordIndex word_index,
      WordIndex universal_word_index) {
    model_index_map_[model_num][word_index] = universal_word_index;
  }

private:
  std::vector<std::vector<WordIndex> > model_index_map_;
};

} // namespace interpolate
} // namespace lm

#endif // LM_INTERPOLATE_UNIVERSAL_VOCAB_H
