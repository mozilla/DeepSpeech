#ifndef LM_INTERPOLATE_NORMALIZE_H
#define LM_INTERPOLATE_NORMALIZE_H

#include "util/fixed_array.hh"

/* Pass 2:
 * - Multiply backoff weights by the backed off probabilities from pass 1.
 * - Compute the normalization factor Z.
 * - Send Z to the next highest order.
 * - Rewind and divide by Z.
 */

namespace util { namespace stream {
class ChainPositions;
class Chains;
}} // namespaces

namespace lm { namespace interpolate {

struct InterpolateInfo;

void Normalize(
    const InterpolateInfo &info,
    // Input full models for backoffs.  Assumes that renumbering has been done. Suffix order.
    util::FixedArray<util::stream::ChainPositions> &models_by_order,
    // Input PartialProbGamma from MergeProbabilities. Context order.
    util::stream::Chains &merged_probabilities,
    // Output NGram<float> with normalized probabilities. Context order.
    util::stream::Chains &probabilities_out,
    // Output bare floats with backoffs.  Note backoffs.size() == order - 1.  Suffix order.
    util::stream::Chains &backoffs_out);

}} // namespaces

#endif // LM_INTERPOLATE_NORMALIZE_H
