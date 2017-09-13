#ifndef LM_INTERPOLATE_TUNE_WEIGHTS_H
#define LM_INTERPOLATE_TUNE_WEIGHTS_H

#include "util/string_piece.hh"

#include <vector>

namespace lm { namespace interpolate {
struct InstancesConfig;

// Run a tuning loop, producing weights as output.
void TuneWeights(int tune_file, const std::vector<StringPiece> &model_names, const InstancesConfig &config, std::vector<float> &weights);

}} // namespaces
#endif // LM_INTERPOLATE_TUNE_WEIGHTS_H
