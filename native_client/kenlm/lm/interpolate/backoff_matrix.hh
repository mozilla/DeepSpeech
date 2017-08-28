#ifndef LM_INTERPOLATE_BACKOFF_MATRIX_H
#define LM_INTERPOLATE_BACKOFF_MATRIX_H

#include <cstddef>
#include <vector>

namespace lm { namespace interpolate {

class BackoffMatrix {
  public:
    BackoffMatrix(std::size_t num_models, std::size_t max_order)
      : max_order_(max_order), backing_(num_models * max_order) {}

    float &Backoff(std::size_t model, std::size_t order_minus_1) {
      return backing_[model * max_order_ + order_minus_1];
    }

    float Backoff(std::size_t model, std::size_t order_minus_1) const {
      return backing_[model * max_order_ + order_minus_1];
    }

  private:
    const std::size_t max_order_;
    std::vector<float> backing_;
};

}} // namespaces

#endif // LM_INTERPOLATE_BACKOFF_MATRIX_H
