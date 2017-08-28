#ifndef KENLM_INTERPOLATE_INTERPOLATE_INFO_H
#define KENLM_INTERPOLATE_INTERPOLATE_INFO_H

#include <cstddef>
#include <vector>
#include <stdint.h>

namespace lm {
namespace interpolate {

/**
 * Stores relevant info for interpolating several language models, for use
 * during the three-pass offline log-linear interpolation algorithm.
 */
struct InterpolateInfo {
  /**
   * @return the number of models being interpolated
   */
  std::size_t Models() const {
    return orders.size();
  }

  /**
   * The lambda (interpolation weight) for each model.
   */
  std::vector<float> lambdas;

  /**
   * The maximum ngram order for each model.
   */
  std::vector<uint8_t> orders;
};
}
}
#endif
