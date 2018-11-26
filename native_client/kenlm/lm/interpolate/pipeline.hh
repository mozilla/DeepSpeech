#ifndef LM_INTERPOLATE_PIPELINE_H
#define LM_INTERPOLATE_PIPELINE_H

#include "lm/common/model_buffer.hh"
#include "util/fixed_array.hh"
#include "util/stream/config.hh"

#include <cstddef>
#include <string>

namespace lm { namespace interpolate {

struct Config {
  std::vector<float> lambdas;
  util::stream::SortConfig sort;
  std::size_t BufferSize() const { return sort.buffer_size; }
};

void Pipeline(util::FixedArray<ModelBuffer> &models, const Config &config, int write_file);

}} // namespaces
#endif // LM_INTERPOLATE_PIPELINE_H
