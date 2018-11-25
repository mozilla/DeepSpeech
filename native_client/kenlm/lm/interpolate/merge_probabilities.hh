#ifndef LM_INTERPOLATE_MERGE_PROBABILITIES_H
#define LM_INTERPOLATE_MERGE_PROBABILITIES_H

#include "lm/common/ngram.hh"
#include "lm/interpolate/bounded_sequence_encoding.hh"
#include "util/fixed_array.hh"
#include "util/stream/multi_stream.hh"

#include <stdint.h>

namespace lm {
namespace interpolate {

struct InterpolateInfo;

/**
 * Make the encoding of backoff values for a given order.  This stores values
 * in [PartialProbGamma::FromBegin(), PartialProbGamma::FromEnd())
 */
BoundedSequenceEncoding MakeEncoder(const InterpolateInfo &info, uint8_t order);

/**
 * The first pass for the offline log-linear interpolation algorithm. This
 * reads K **suffix-ordered** streams for each model, for each order, of
 * ngram records (ngram-id, prob, backoff). It further assumes that the
 * ngram-ids have been unified over all of the stream inputs.
 *
 * Its output is records of (ngram-id, prob-prod, backoff-level,
 * backoff-level, ...) where the backoff-levels (of which there are K) are
 * the context length (0 for unigrams) that the corresponding model had to
 * back off to in order to obtain a probability for that ngram-id. Each of
 * these streams is terminated with a record whose ngram-id is all
 * maximum-integers for simplicity in implementation here.
 *
 * @param model_by_order An array of length N (max_i N_i) containing at
 *  the ChainPositions for the streams for order (i + 1).
 * The Rus attached to output chains for each order (of length K)
 */
class MergeProbabilities {
  public:
    MergeProbabilities(const InterpolateInfo &info, util::FixedArray<util::stream::ChainPositions> &models_by_order)
      : info_(info), models_by_order_(models_by_order) {}

    void Run(const util::stream::ChainPositions &outputs);

  private:
    const InterpolateInfo &info_;
    util::FixedArray<util::stream::ChainPositions> &models_by_order_;
};

/**
 * This class represents the output payload for this pass, which consists
 * of an ngram-id, a probability, and then a vector of orders from which
 * each of the component models backed off to for this ngram, encoded
 * using the BoundedSequenceEncoding class.
 */
class PartialProbGamma : public lm::NGramHeader {
public:
  PartialProbGamma(std::size_t order, std::size_t backoff_bytes)
      : lm::NGramHeader(NULL, order), backoff_bytes_(backoff_bytes) {
    // nothing
  }

  std::size_t TotalSize() const {
    return sizeof(WordIndex) * Order() + sizeof(After) + backoff_bytes_;
  }

  // TODO: cache bounded sequence encoding in the pipeline?
  static std::size_t TotalSize(const InterpolateInfo &info, uint8_t order) {
    return sizeof(WordIndex) * order + sizeof(After) + MakeEncoder(info, order).EncodedLength();
  }

  float &Prob() { return Pay().prob; }
  float Prob() const { return Pay().prob; }

  float &LowerProb() { return Pay().lower_prob; }
  float LowerProb() const { return Pay().lower_prob; }

  const uint8_t *FromBegin() const { return Pay().from; }
  uint8_t *FromBegin() { return Pay().from; }

private:
  struct After {
    // Note that backoff_and_normalize assumes this comes first.
    float prob;
    float lower_prob;
    uint8_t from[];
  };
  const After &Pay() const { return *reinterpret_cast<const After *>(end()); }
  After &Pay() { return *reinterpret_cast<After*>(end()); }

  std::size_t backoff_bytes_;
};

}} // namespaces
#endif // LM_INTERPOLATE_MERGE_PROBABILITIES_H
