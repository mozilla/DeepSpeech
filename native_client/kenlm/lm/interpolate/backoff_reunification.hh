#ifndef KENLM_INTERPOLATE_BACKOFF_REUNIFICATION_
#define KENLM_INTERPOLATE_BACKOFF_REUNIFICATION_

#include "util/stream/stream.hh"
#include "util/stream/multi_stream.hh"

namespace lm {
namespace interpolate {

/**
 * The third pass for the offline log-linear interpolation algorithm. This
 * reads **suffix-ordered** probability values (ngram-id, float) and
 * **suffix-ordered** backoff values (float) and writes the merged contents
 * to the output.
 *
 * @param prob_pos The chain position for each order from which to read
 *  the probability values
 * @param boff_pos The chain position for each order from which to read
 *  the backoff values
 * @param output_chains The output chains for each order
 */
void ReunifyBackoff(util::stream::ChainPositions &prob_pos,
                    util::stream::ChainPositions &boff_pos,
                    util::stream::Chains &output_chains);
}
}
#endif
