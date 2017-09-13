#ifndef LM_VALUE_BUILD_H
#define LM_VALUE_BUILD_H

#include "lm/weights.hh"
#include "lm/word_index.hh"
#include "util/bit_packing.hh"

#include <vector>

namespace lm {
namespace ngram {

struct Config;
struct BackoffValue;
struct RestValue;

class NoRestBuild {
  public:
    typedef BackoffValue Value;

    NoRestBuild() {}

    void SetRest(const WordIndex *, unsigned int, const Prob &/*prob*/) const {}
    void SetRest(const WordIndex *, unsigned int, const ProbBackoff &) const {}

    template <class Second> bool MarkExtends(ProbBackoff &weights, const Second &) const {
      util::UnsetSign(weights.prob);
      return false;
    }

    // Probing doesn't need to go back to unigram.
    const static bool kMarkEvenLower = false;
};

class MaxRestBuild {
  public:
    typedef RestValue Value;

    MaxRestBuild() {}

    void SetRest(const WordIndex *, unsigned int, const Prob &/*prob*/) const {}
    void SetRest(const WordIndex *, unsigned int, RestWeights &weights) const {
      weights.rest = weights.prob;
      util::SetSign(weights.rest);
    }

    bool MarkExtends(RestWeights &weights, const RestWeights &to) const {
      util::UnsetSign(weights.prob);
      if (weights.rest >= to.rest) return false;
      weights.rest = to.rest;
      return true;
    }
    bool MarkExtends(RestWeights &weights, const Prob &to) const {
      util::UnsetSign(weights.prob);
      if (weights.rest >= to.prob) return false;
      weights.rest = to.prob;
      return true;
    }

    // Probing does need to go back to unigram.
    const static bool kMarkEvenLower = true;
};

template <class Model> class LowerRestBuild {
  public:
    typedef RestValue Value;

    LowerRestBuild(const Config &config, unsigned int order, const typename Model::Vocabulary &vocab);

    ~LowerRestBuild();

    void SetRest(const WordIndex *, unsigned int, const Prob &/*prob*/) const {}
    void SetRest(const WordIndex *vocab_ids, unsigned int n, RestWeights &weights) const {
      typename Model::State ignored;
      if (n == 1) {
        weights.rest = unigrams_[*vocab_ids];
      } else {
        weights.rest = models_[n-2]->FullScoreForgotState(vocab_ids + 1, vocab_ids + n, *vocab_ids, ignored).prob;
      }
    }

    template <class Second> bool MarkExtends(RestWeights &weights, const Second &) const {
      util::UnsetSign(weights.prob);
      return false;
    }

    const static bool kMarkEvenLower = false;

    std::vector<float> unigrams_;

    std::vector<const Model*> models_;
};

} // namespace ngram
} // namespace lm

#endif // LM_VALUE_BUILD_H
