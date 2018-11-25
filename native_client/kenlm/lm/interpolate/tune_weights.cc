#include "lm/interpolate/tune_weights.hh"

#include "lm/interpolate/tune_derivatives.hh"
#include "lm/interpolate/tune_instances.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas" // Older gcc doesn't have "-Wunused-local-typedefs" and complains.
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#include <boost/program_options.hpp>

#include <iostream>

namespace lm { namespace interpolate {
void TuneWeights(int tune_file, const std::vector<StringPiece> &model_names, const InstancesConfig &config, std::vector<float> &weights_out) {
  Instances instances(tune_file, model_names, config);
  Vector weights = Vector::Constant(model_names.size(), 1.0 / model_names.size());
  Vector gradient;
  Matrix hessian;
  for (std::size_t iteration = 0; iteration < 10 /*TODO fancy stopping criteria */; ++iteration) {
    std::cerr << "Iteration " << iteration << ": weights =";
    for (Vector::Index i = 0; i < weights.rows(); ++i) {
      std::cerr << ' ' << weights(i);
    }
    std::cerr << std::endl;
    std::cerr << "Perplexity = " << Derivatives(instances, weights, gradient, hessian) << std::endl;
    // TODO: 1.0 step size was too big and it kept getting unstable.  More math.
    weights -= 0.7 * hessian.inverse() * gradient;
  }
  weights_out.assign(weights.data(), weights.data() + weights.size());
}
}} // namespaces
