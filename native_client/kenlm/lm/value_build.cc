#include "lm/value_build.hh"

#include "lm/model.hh"
#include "lm/read_arpa.hh"

namespace lm {
namespace ngram {

template <class Model> LowerRestBuild<Model>::LowerRestBuild(const Config &config, unsigned int order, const typename Model::Vocabulary &vocab) {
  UTIL_THROW_IF(config.rest_lower_files.size() != order - 1, ConfigException, "This model has order " << order << " so there should be " << (order - 1) << " lower-order models for rest cost purposes.");
  Config for_lower = config;
  for_lower.write_mmap = NULL;
  for_lower.rest_lower_files.clear();

  // Unigram models aren't supported, so this is a custom loader.
  // TODO: optimize the unigram loading?
  {
    util::FilePiece uni(config.rest_lower_files[0].c_str());
    std::vector<uint64_t> number;
    ReadARPACounts(uni, number);
    UTIL_THROW_IF(number.size() != 1, FormatLoadException, "Expected the unigram model to have order 1, not " << number.size());
    ReadNGramHeader(uni, 1);
    unigrams_.resize(number[0]);
    unigrams_[0] = config.unknown_missing_logprob;
    PositiveProbWarn warn;
    for (uint64_t i = 0; i < number[0]; ++i) {
      WordIndex w;
      Prob entry;
      ReadNGram(uni, 1, vocab, &w, entry, warn);
      unigrams_[w] = entry.prob;
    }
  }

  try {
    for (unsigned int i = 2; i < order; ++i) {
      models_.push_back(new Model(config.rest_lower_files[i - 1].c_str(), for_lower));
      UTIL_THROW_IF(models_.back()->Order() != i, FormatLoadException, "Lower order file " << config.rest_lower_files[i-1] << " should have order " << i);
    }
  } catch (...) {
    for (typename std::vector<const Model*>::const_iterator i = models_.begin(); i != models_.end(); ++i) {
      delete *i;
    }
    models_.clear();
    throw;
  }

  // TODO: force/check same vocab.
}

template <class Model> LowerRestBuild<Model>::~LowerRestBuild() {
  for (typename std::vector<const Model*>::const_iterator i = models_.begin(); i != models_.end(); ++i) {
    delete *i;
  }
}

template class LowerRestBuild<ProbingModel>;

} // namespace ngram
} // namespace lm
