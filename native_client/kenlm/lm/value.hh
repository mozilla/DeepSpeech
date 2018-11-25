#ifndef LM_VALUE_H
#define LM_VALUE_H

#include "lm/config.hh"
#include "lm/model_type.hh"
#include "lm/value_build.hh"
#include "lm/weights.hh"
#include "util/bit_packing.hh"

#include <stdint.h>

namespace lm {
namespace ngram {

// Template proxy for probing unigrams and middle.
template <class Weights> class GenericProbingProxy {
  public:
    explicit GenericProbingProxy(const Weights &to) : to_(&to) {}

    GenericProbingProxy() : to_(0) {}

    bool Found() const { return to_ != 0; }

    float Prob() const {
      util::FloatEnc enc;
      enc.f = to_->prob;
      enc.i |= util::kSignBit;
      return enc.f;
    }

    float Backoff() const { return to_->backoff; }

    bool IndependentLeft() const {
      util::FloatEnc enc;
      enc.f = to_->prob;
      return enc.i & util::kSignBit;
    }

  protected:
    const Weights *to_;
};

// Basic proxy for trie unigrams.
template <class Weights> class GenericTrieUnigramProxy {
  public:
    explicit GenericTrieUnigramProxy(const Weights &to) : to_(&to) {}

    GenericTrieUnigramProxy() : to_(0) {}

    bool Found() const { return to_ != 0; }
    float Prob() const { return to_->prob; }
    float Backoff() const { return to_->backoff; }
    float Rest() const { return Prob(); }

  protected:
    const Weights *to_;
};

struct BackoffValue {
  typedef ProbBackoff Weights;
  static const ModelType kProbingModelType = PROBING;

  class ProbingProxy : public GenericProbingProxy<Weights> {
    public:
      explicit ProbingProxy(const Weights &to) : GenericProbingProxy<Weights>(to) {}
      ProbingProxy() {}
      float Rest() const { return Prob(); }
  };

  class TrieUnigramProxy : public GenericTrieUnigramProxy<Weights> {
    public:
      explicit TrieUnigramProxy(const Weights &to) : GenericTrieUnigramProxy<Weights>(to) {}
      TrieUnigramProxy() {}
      float Rest() const { return Prob(); }
  };

  struct ProbingEntry {
    typedef uint64_t Key;
    typedef Weights Value;
    uint64_t key;
    ProbBackoff value;
    uint64_t GetKey() const { return key; }
  };

  struct TrieUnigramValue {
    Weights weights;
    uint64_t next;
    uint64_t Next() const { return next; }
  };

  const static bool kDifferentRest = false;

  template <class Model, class C> void Callback(const Config &, unsigned int, typename Model::Vocabulary &, C &callback) {
    NoRestBuild build;
    callback(build);
  }
};

struct RestValue {
  typedef RestWeights Weights;
  static const ModelType kProbingModelType = REST_PROBING;

  class ProbingProxy : public GenericProbingProxy<RestWeights> {
    public:
      explicit ProbingProxy(const Weights &to) : GenericProbingProxy<RestWeights>(to) {}
      ProbingProxy() {}
      float Rest() const { return to_->rest; }
  };

  class TrieUnigramProxy : public GenericTrieUnigramProxy<Weights> {
    public:
      explicit TrieUnigramProxy(const Weights &to) : GenericTrieUnigramProxy<Weights>(to) {}
      TrieUnigramProxy() {}
      float Rest() const { return to_->rest; }
  };

// gcc 4.1 doesn't properly back dependent types :-(.
#pragma pack(push)
#pragma pack(4)
  struct ProbingEntry {
    typedef uint64_t Key;
    typedef Weights Value;
    Key key;
    Value value;
    Key GetKey() const { return key; }
  };

  struct TrieUnigramValue {
    Weights weights;
    uint64_t next;
    uint64_t Next() const { return next; }
  };
#pragma pack(pop)

  const static bool kDifferentRest = true;

  template <class Model, class C> void Callback(const Config &config, unsigned int order, typename Model::Vocabulary &vocab, C &callback) {
    switch (config.rest_function) {
      case Config::REST_MAX:
        {
          MaxRestBuild build;
          callback(build);
        }
        break;
      case Config::REST_LOWER:
        {
          LowerRestBuild<Model> build(config, order, vocab);
          callback(build);
        }
        break;
    }
  }
};

} // namespace ngram
} // namespace lm

#endif // LM_VALUE_H
