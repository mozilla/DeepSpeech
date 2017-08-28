#ifndef LM_INTERPOLATE_TUNE_INSTANCE_H
#define LM_INTERPOLATE_TUNE_INSTANCE_H

#include "lm/interpolate/tune_matrix.hh"
#include "lm/word_index.hh"
#include "util/scoped.hh"
#include "util/stream/config.hh"
#include "util/string_piece.hh"

#include <boost/optional.hpp>

#include <vector>

namespace util { namespace stream {
class Chain;
class FileBuffer;
}} // namespaces

namespace lm { namespace interpolate {

typedef uint32_t InstanceIndex;
typedef uint32_t ModelIndex;

struct Extension {
  // Which tuning instance does this belong to?
  InstanceIndex instance;
  WordIndex word;
  ModelIndex model;
  // ln p_{model} (word | context(instance))
  float ln_prob;

  bool operator<(const Extension &other) const;
};

class ExtensionsFirstIteration;

struct InstancesConfig {
  // For batching the model reads.  This is per order.
  std::size_t model_read_chain_mem;
  // This is being sorted, make it larger.
  std::size_t extension_write_chain_mem;
  std::size_t lazy_memory;
  util::stream::SortConfig sort;
};

class Instances {
  private:
    typedef Eigen::Matrix<Accum, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> BackoffMatrix;

  public:
    Instances(int tune_file, const std::vector<StringPiece> &model_names, const InstancesConfig &config);

    // For destruction of forward-declared classes.
    ~Instances();

    // Full backoff from unigram for each model.
    typedef BackoffMatrix::ConstRowXpr FullBackoffs;
    FullBackoffs LNBackoffs(InstanceIndex instance) const {
      return ln_backoffs_.row(instance);
    }

    InstanceIndex NumInstances() const { return ln_backoffs_.rows(); }

    const Vector &CorrectGradientTerm() const { return neg_ln_correct_sum_; }

    const Matrix &LNUnigrams() const { return ln_unigrams_; }

    // Entry size to use to configure the chain (since in practice order is needed).
    std::size_t ReadExtensionsEntrySize() const;
    void ReadExtensions(util::stream::Chain &chain);

    // Vocab id of the beginning of sentence.  Used to ignore it for normalization.
    WordIndex BOS() const { return bos_; }

  private:
    // Allow the derivatives test to get access.
    friend class MockInstances;
    Instances();

    // backoffs_(instance, model) is the backoff all the way to unigrams.
    BackoffMatrix ln_backoffs_;

    // neg_correct_sum_(model) = -\sum_{instances} ln p_{model}(correct(instance) | context(instance)).
    // This appears as a term in the gradient.
    Vector neg_ln_correct_sum_;

    // ln_unigrams_(word, model) = ln p_{model}(word).
    Matrix ln_unigrams_;

    // This is the source of data for the first iteration.
    util::scoped_ptr<ExtensionsFirstIteration> extensions_first_;

    // Source of data for subsequent iterations.  This contains already-sorted data.
    util::scoped_ptr<util::stream::FileBuffer> extensions_subsequent_;

    WordIndex bos_;

    std::string temp_prefix_;
};

}} // namespaces
#endif // LM_INTERPOLATE_TUNE_INSTANCE_H
