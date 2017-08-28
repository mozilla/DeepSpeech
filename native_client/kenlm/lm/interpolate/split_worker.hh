#ifndef KENLM_INTERPOLATE_SPLIT_WORKER_H_
#define KENLM_INTERPOLATE_SPLIT_WORKER_H_

#include "util/stream/chain.hh"
#include "util/stream/stream.hh"

namespace lm {
namespace interpolate {

class SplitWorker {
  public:
    /**
     * Constructs a split worker for a particular order. It writes the
     * split-off backoff values to the backoff chain and the ngram id and
     * probability to the sort chain for each ngram in the input.
     */
    SplitWorker(std::size_t order, util::stream::Chain &backoff_chain,
                util::stream::Chain &sort_chain);

    /**
     * The callback invoked to handle the input from the ngram intermediate
     * files.
     */
    void Run(const util::stream::ChainPosition& position);

  private:
    /**
     * The ngram order we are reading/writing for.
     */
    std::size_t order_;

    /**
     * The stream to write to for the backoff values.
     */
    util::stream::Stream backoff_input_;

    /**
     * The stream to write to for the ngram id + probability values.
     */
    util::stream::Stream sort_input_;
};
}
}
#endif
