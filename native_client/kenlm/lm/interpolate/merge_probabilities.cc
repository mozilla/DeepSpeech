#include "lm/interpolate/merge_probabilities.hh"
#include "lm/common/ngram_stream.hh"
#include "lm/interpolate/bounded_sequence_encoding.hh"
#include "lm/interpolate/interpolate_info.hh"

#include <algorithm>
#include <limits>
#include <numeric>

namespace lm {
namespace interpolate {

/**
 * Helper to generate the BoundedSequenceEncoding used for writing the
 * from values.
 */
BoundedSequenceEncoding MakeEncoder(const InterpolateInfo &info, uint8_t order) {
  util::FixedArray<uint8_t> max_orders(info.orders.size());
  for (std::size_t i = 0; i < info.orders.size(); ++i) {
    max_orders.push_back(std::min(order, info.orders[i]));
  }
  return BoundedSequenceEncoding(max_orders.begin(), max_orders.end());
}

namespace {

/**
 * A simple wrapper class that holds information needed to read and write
 * the ngrams of a particular order. This class has the memory needed to
 * buffer the data needed for the recursive process of computing the
 * probabilities and "from" values for each component model.
 *
 * "From" values indicate, for each model, what order (as an index, so -1)
 * was backed off to in order to arrive at a probability. For example, if a
 * 5-gram model (order index 4) backed off twice, we would write a 2.
 */
class NGramHandler {
public:
  NGramHandler(uint8_t order, const InterpolateInfo &ifo,
               util::FixedArray<util::stream::ChainPositions> &models_by_order)
      : info(ifo),
        encoder(MakeEncoder(info, order)),
        out_record(order, encoder.EncodedLength()) {
    std::size_t count_has_order = 0;
    for (std::size_t i = 0; i < models_by_order.size(); ++i) {
      count_has_order += (models_by_order[i].size() >= order);
    }
    inputs_.Init(count_has_order);
    for (std::size_t i = 0; i < models_by_order.size(); ++i) {
      if (models_by_order[i].size() < order)
        continue;
      inputs_.push_back(models_by_order[i][order - 1]);
      if (inputs_.back()) {
        active_.resize(active_.size() + 1);
        active_.back().model = i;
        active_.back().stream = &inputs_.back();
      }
    }

    // have to init outside since NGramStreams doesn't forward to
    // GenericStreams ctor given a ChainPositions

    probs.Init(info.Models());
    from.Init(info.Models());
    for (std::size_t i = 0; i < info.Models(); ++i) {
      probs.push_back(0.0);
      from.push_back(0);
    }
  }

  struct StreamIndex {
    NGramStream<ProbBackoff> *stream;
    NGramStream<ProbBackoff> &Stream() { return *stream; }
    std::size_t model;
  };

  std::size_t ActiveSize() const {
    return active_.size();
  }

  /**
   * @return the input stream for a particular model that corresponds to
   * this ngram order
   */
  StreamIndex &operator[](std::size_t idx) {
    return active_[idx];
  }

  void erase(std::size_t idx) {
    active_.erase(active_.begin() + idx);
  }

  const InterpolateInfo &info;
  BoundedSequenceEncoding encoder;
  PartialProbGamma out_record;
  util::FixedArray<float> probs;
  util::FixedArray<uint8_t> from;

private:
  std::vector<StreamIndex> active_;
  NGramStreams<ProbBackoff> inputs_;
};

/**
 * A collection of NGramHandlers.
 */
class NGramHandlers : public util::FixedArray<NGramHandler> {
public:
  explicit NGramHandlers(std::size_t num)
      : util::FixedArray<NGramHandler>(num) {
  }

  void push_back(
      std::size_t order, const InterpolateInfo &info,
      util::FixedArray<util::stream::ChainPositions> &models_by_order) {
    new (end()) NGramHandler(order, info, models_by_order);
    Constructed();
  }
};

/**
 * The recursive helper function that computes probability and "from"
 * values for all ngrams matching a particular suffix.
 *
 * The current order can be computed as the suffix length + 1. Note that
 * the suffix could be empty (suffix_begin == suffix_end == NULL), in which
 * case we are handling unigrams with the UNK token as the fallback
 * probability.
 *
 * @param handlers The full collection of handlers
 * @param suffix_begin A start iterator for the suffix
 * @param suffix_end An end iterator for the suffix
 * @param fallback_probs The probabilities of this ngram if we need to
 *  back off (that is, the probability of the suffix)
 * @param fallback_from The order that the corresponding fallback
 *  probability in the fallback_probs is from
 * @param combined_fallback interpolated fallback_probs
 * @param outputs The output streams, one for each order
 */
void HandleSuffix(NGramHandlers &handlers, WordIndex *suffix_begin,
                  WordIndex *suffix_end,
                  const util::FixedArray<float> &fallback_probs,
                  const util::FixedArray<uint8_t> &fallback_from,
                  float combined_fallback,
                  util::stream::Streams &outputs) {
  uint8_t order = std::distance(suffix_begin, suffix_end) + 1;
  if (order > outputs.size()) return;

  util::stream::Stream &output = outputs[order - 1];
  NGramHandler &handler = handlers[order - 1];

  while (true) {
    // find the next smallest ngram which matches our suffix
    // TODO: priority queue driven.
    WordIndex *minimum = NULL;
    for (std::size_t i = 0; i < handler.ActiveSize(); ++i) {
      if (!std::equal(suffix_begin, suffix_end, handler[i].Stream()->begin() + 1))
        continue;

      // if we either haven't set a minimum yet or this one is smaller than
      // the minimum we found before, replace it
      WordIndex *last = handler[i].Stream()->begin();
      if (!minimum || *last < *minimum) { minimum = handler[i].Stream()->begin(); }
    }

    // no more ngrams of this order match our suffix, so we're done
    if (!minimum) return;

    handler.out_record.ReBase(output.Get());
    std::copy(minimum, minimum + order, handler.out_record.begin());

    // Default case is having backed off.
    std::copy(fallback_probs.begin(), fallback_probs.end(), handler.probs.begin());
    std::copy(fallback_from.begin(), fallback_from.end(), handler.from.begin());

    for (std::size_t i = 0; i < handler.ActiveSize();) {
      if (std::equal(handler.out_record.begin(), handler.out_record.end(),
                     handler[i].Stream()->begin())) {
        handler.probs[handler[i].model] = handler.info.lambdas[handler[i].model] * handler[i].Stream()->Value().prob;
        handler.from[handler[i].model] = order - 1;
        if (++handler[i].Stream()) {
          ++i;
        } else {
          handler.erase(i);
        }
      } else {
        ++i;
      }
    }
    handler.out_record.Prob() = std::accumulate(handler.probs.begin(), handler.probs.end(), 0.0);
    handler.out_record.LowerProb() = combined_fallback;
    handler.encoder.Encode(handler.from.begin(),
                           handler.out_record.FromBegin());

    // we've handled this particular ngram, so now recurse to the higher
    // order using the current ngram as the suffix
    HandleSuffix(handlers, handler.out_record.begin(), handler.out_record.end(),
                 handler.probs, handler.from, handler.out_record.Prob(), outputs);
    // consume the output
    ++output;
  }
}

/**
 * Kicks off the recursion for computing the probabilities and "from"
 * values for each ngram order. We begin by handling the UNK token that
 * should be at the front of each of the unigram input streams. This is
 * then output to the stream and it is used as the fallback for handling
 * our unigram case, the unigram used as the fallback for the bigram case,
 * etc.
 */
void HandleNGrams(NGramHandlers &handlers, util::stream::Streams &outputs) {
  PartialProbGamma unk_record(1, 0);
  // First: populate the unk probabilities by reading the first unigram
  // from each stream
  util::FixedArray<float> unk_probs(handlers[0].info.Models());

  // start by populating the ngram id from the first stream
  lm::NGram<ProbBackoff> ngram = *handlers[0][0].Stream();
  unk_record.ReBase(outputs[0].Get());
  std::copy(ngram.begin(), ngram.end(), unk_record.begin());
  unk_record.Prob() = 0;

  // then populate the probabilities into unk_probs while "multiply" the
  // model probabilities together into the unk record
  //
  // note that from doesn't need to be set for unigrams
  assert(handlers[0].ActiveSize() == handlers[0].info.Models());
  for (std::size_t i = 0; i < handlers[0].info.Models();) {
    ngram = *handlers[0][i].Stream();
    unk_probs.push_back(handlers[0].info.lambdas[i] * ngram.Value().prob);
    unk_record.Prob() += unk_probs[i];
    assert(*ngram.begin() == kUNK);
    if (++handlers[0][i].Stream()) {
      ++i;
    } else {
      handlers[0].erase(i);
    }
  }
  float unk_combined = unk_record.Prob();
  unk_record.LowerProb() = unk_combined;
  // flush the unk output record
  ++outputs[0];

  // Then, begin outputting everything in lexicographic order: first we'll
  // get the unigram then the first bigram with that context, then the
  // first trigram with that bigram context, etc., until we exhaust all of
  // the ngrams, then all of the (n-1)grams, etc.
  //
  // This function is the "root" of this recursive process.
  util::FixedArray<uint8_t> unk_from(handlers[0].info.Models());
  for (std::size_t i = 0; i < handlers[0].info.Models(); ++i) {
    unk_from.push_back(0);
  }

  // the two nulls are to encode that our "fallback" word is the "0-gram"
  // case, e.g. we "backed off" to UNK
  // TODO: stop generating vocab ids and LowerProb for unigrams.
  HandleSuffix(handlers, NULL, NULL, unk_probs, unk_from, unk_combined, outputs);

  // Verify we reached the end.  And poison!
  for (std::size_t i = 0; i < handlers.size(); ++i) {
    UTIL_THROW_IF2(handlers[i].ActiveSize(),
                     "MergeProbabilities did not exhaust all ngram streams");
    outputs[i].Poison();
  }
}
} // namespace

void MergeProbabilities::Run(const util::stream::ChainPositions &output_pos) {
  NGramHandlers handlers(output_pos.size());
  for (std::size_t i = 0; i < output_pos.size(); ++i) {
    handlers.push_back(i + 1, info_, models_by_order_);
  }

  util::stream::Streams outputs(output_pos);
  HandleNGrams(handlers, outputs);
}

}} // namespaces
