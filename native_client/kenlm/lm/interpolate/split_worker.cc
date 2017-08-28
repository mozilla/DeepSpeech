#include "lm/interpolate/split_worker.hh"
#include "lm/common/ngram.hh"

namespace lm {
namespace interpolate {

SplitWorker::SplitWorker(std::size_t order, util::stream::Chain &backoff_chain,
                         util::stream::Chain &sort_chain)
    : order_(order) {
  backoff_chain >> backoff_input_;
  sort_chain >> sort_input_;
}

void SplitWorker::Run(const util::stream::ChainPosition &position) {
  // input: ngram record (id, prob, and backoff)
  // output: a float to the backoff_input stream
  //         an ngram id and a float to the sort_input stream
  for (util::stream::Stream stream(position); stream; ++stream) {
    NGram<ProbBackoff> ngram(stream.Get(), order_);

    // write id and prob to the sort stream
    float prob = ngram.Value().prob;
    lm::WordIndex *out = reinterpret_cast<lm::WordIndex *>(sort_input_.Get());
    for (const lm::WordIndex *it = ngram.begin(); it != ngram.end(); ++it) {
      *out++ = *it;
    }
    *reinterpret_cast<float *>(out) = prob;
    ++sort_input_;

    // write backoff to the backoff output stream
    float boff = ngram.Value().backoff;
    *reinterpret_cast<float *>(backoff_input_.Get()) = boff;
    ++backoff_input_;
  }
  sort_input_.Poison();
  backoff_input_.Poison();
}

}
}
