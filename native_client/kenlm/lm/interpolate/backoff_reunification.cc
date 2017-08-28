#include "lm/interpolate/backoff_reunification.hh"
#include "lm/common/model_buffer.hh"
#include "lm/common/ngram_stream.hh"
#include "lm/common/ngram.hh"
#include "lm/common/compare.hh"

#include <algorithm>
#include <cassert>

namespace lm {
namespace interpolate {

namespace {
class MergeWorker {
public:
  MergeWorker(std::size_t order, const util::stream::ChainPosition &prob_pos,
              const util::stream::ChainPosition &boff_pos)
      : order_(order), prob_pos_(prob_pos), boff_pos_(boff_pos) {
    // nothing
  }

  void Run(const util::stream::ChainPosition &position) {
    lm::NGramStream<ProbBackoff> stream(position);

    lm::NGramStream<float> prob_input(prob_pos_);
    util::stream::Stream boff_input(boff_pos_);
    for (; prob_input && boff_input; ++prob_input, ++boff_input, ++stream) {
      std::copy(prob_input->begin(), prob_input->end(), stream->begin());
      stream->Value().prob = std::min(0.0f, prob_input->Value());
      stream->Value().backoff = *reinterpret_cast<float *>(boff_input.Get());
    }
    UTIL_THROW_IF2(prob_input || boff_input,
                   "Streams were not the same size during merging");
    stream.Poison();
  }

private:
  std::size_t order_;
  util::stream::ChainPosition prob_pos_;
  util::stream::ChainPosition boff_pos_;
};
}

// Since we are *adding* something to the output chain here, we pass in the
// chain itself so that we can safely add a new step to the chain without
// creating a deadlock situation (since creating a new ChainPosition will
// make a new input/output pair---we want that position to be created
// *here*, not before).
void ReunifyBackoff(util::stream::ChainPositions &prob_pos,
                    util::stream::ChainPositions &boff_pos,
                    util::stream::Chains &output_chains) {
  assert(prob_pos.size() == boff_pos.size());

  for (size_t i = 0; i < prob_pos.size(); ++i)
    output_chains[i] >> MergeWorker(i + 1, prob_pos[i], boff_pos[i]);
}
}
}
