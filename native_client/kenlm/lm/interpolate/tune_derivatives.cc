#include "lm/interpolate/tune_derivatives.hh"

#include "lm/interpolate/tune_instances.hh"
#include "lm/interpolate/tune_matrix.hh"
#include "util/stream/chain.hh"
#include "util/stream/typed_stream.hh"

#include <Eigen/Core>

namespace lm { namespace interpolate {

Accum Derivatives(Instances &in, const Vector &weights, Vector &gradient, Matrix &hessian) {
  gradient = in.CorrectGradientTerm();
  hessian = Matrix::Zero(weights.rows(), weights.rows());

  // TODO: loop instead to force low-memory evaluation?
  // Compute p_I(x)*Z_{\epsilon} i.e. the unnormalized probabilities
  Vector weighted_uni((in.LNUnigrams() * weights).array().exp());
  // Even -inf doesn't work for <s> because weights can be negative.  Manually set it to zero.
  weighted_uni(in.BOS()) = 0.0;
  Accum Z_epsilon = weighted_uni.sum();
  // unigram_cross(i) = \sum_{all x} p_I(x) ln p_i(x)
  Vector unigram_cross(in.LNUnigrams().transpose() * weighted_uni / Z_epsilon);

  Accum sum_B_I = 0.0;
  Accum sum_ln_Z_context = 0.0;

  // Temporaries used each cycle of the loop.
  Matrix convolve;
  Vector full_cross;
  Matrix hessian_missing_Z_context;
  // Backed off ln p_i(x)B_i(context)
  Vector ln_p_i_backed;
  // Full ln p_i(x | context)
  Vector ln_p_i_full;

  // TODO make configurable memory size.
  util::stream::Chain chain(util::stream::ChainConfig(in.ReadExtensionsEntrySize(), 2, 64 << 20));
  chain.ActivateProgress();
  in.ReadExtensions(chain);
  util::stream::TypedStream<Extension> extensions(chain.Add());
  chain >> util::stream::kRecycle;

  // Loop over instances (words in the tuning data).
  for (InstanceIndex n = 0; n < in.NumInstances(); ++n) {
    assert(extensions);
    Accum weighted_backoffs = exp(in.LNBackoffs(n).dot(weights));

    // Compute \sum_{x: model does not back off to unigram} p_I(x)Z(epsilon)
    Accum unnormalized_sum_x_p_I = 0.0;
    // Compute \sum_{x: model does not back off to unigram} p_I(x | context)Z(context)
    Accum unnormalized_sum_x_p_I_full = 0.0;

    // This should be divided by Z_context then added to the Hessian.
    hessian_missing_Z_context = Matrix::Zero(weights.rows(), weights.rows());

    full_cross = Vector::Zero(weights.rows());

    // Loop over words within an instance for which extension exists.  An extension happens when any model matches more than a unigram in the tuning instance.
    while (extensions && extensions->instance == n) {
      const WordIndex word = extensions->word;
      unnormalized_sum_x_p_I += weighted_uni(word);

      ln_p_i_backed = in.LNUnigrams().row(word) + in.LNBackoffs(n);

      // Calculate ln_p_i_full(i) = ln p_i(word | context) by filling in unigrams then overwriting with extensions.
      ln_p_i_full = ln_p_i_backed;
      // Loop over all models that have an extension for the same word namely p_i(word | context) matches at least a bigram.
      for (; extensions && extensions->word == word && extensions->instance == n; ++extensions) {
        ln_p_i_full(extensions->model) = extensions->ln_prob;
      }

      // This is the weighted product of probabilities.  In other words, p_I(word | context) * Z(context) = exp(\sum_i w_i * p_i(word | context)).
      Accum weighted = exp(ln_p_i_full.dot(weights));
      unnormalized_sum_x_p_I_full += weighted;

      // These aren't normalized by Z_context (happens later)
      full_cross.noalias() +=
        weighted * ln_p_i_full
        - weighted_uni(word) * weighted_backoffs /* we'll divide by Z_context later to form B_I */ * in.LNUnigrams().row(word).transpose();

      // This will get multiplied by Z_context then added to the Hessian.
      hessian_missing_Z_context.noalias() +=
        // Replacement terms.
        weighted * ln_p_i_full * ln_p_i_full.transpose()
        // Presumed unigrams.  Z_epsilon * weighted_backoffs will turn into B_I once all of this is divided by Z_context.
        - weighted_uni(word) * weighted_backoffs * ln_p_i_backed * ln_p_i_backed.transpose();
    }

    Accum Z_context =
      weighted_backoffs * (Z_epsilon - unnormalized_sum_x_p_I) // Back off and unnormalize the unigrams for which there is no extension.
      + unnormalized_sum_x_p_I_full; // Add the extensions.
    sum_ln_Z_context += log(Z_context);
    Accum B_I = Z_epsilon / Z_context * weighted_backoffs;
    sum_B_I += B_I;

    // This is the gradient term for this instance except for -log p_i(w_n | w_1^{n-1}) which was accounted for as part of neg_correct_sum_.
    // full_cross(i) is \sum_{all x} p_I(x | context) log p_i(x | context)
    // Prior terms excluded dividing by Z_context because it wasn't known at the time.
    full_cross /= Z_context;
    full_cross +=
      // Uncorrected term
      B_I * (in.LNBackoffs(n).transpose() + unigram_cross)
      // Subtract values that should not have been charged.
      - unnormalized_sum_x_p_I / Z_epsilon * B_I * in.LNBackoffs(n).transpose();
    gradient += full_cross;

    convolve = unigram_cross * in.LNBackoffs(n);
    // There's one missing term here, which is independent of context and done at the end.
    hessian.noalias() +=
      // First term of Hessian, assuming all models back off to unigram.
      B_I * (convolve + convolve.transpose() + in.LNBackoffs(n).transpose() * in.LNBackoffs(n))
      // Error in the first term, correcting from unigram to full probabilities.
      + hessian_missing_Z_context / Z_context
      // Second term of Hessian, with correct full probabilities.
      - full_cross * full_cross.transpose();
  }

  for (Matrix::Index x = 0; x < weighted_uni.rows(); ++x) {
    // \sum_{contexts} B_I(context) \sum_x p_I(x) log p_i(x) log p_j(x)
    // TODO can this be optimized?  It's summing over the entire vocab which should be a matrix operation.
    hessian.noalias() += sum_B_I * weighted_uni(x) / Z_epsilon * in.LNUnigrams().row(x).transpose() * in.LNUnigrams().row(x);
  }
  return exp((in.CorrectGradientTerm().dot(weights) + sum_ln_Z_context) / static_cast<double>(in.NumInstances()));
}

}} // namespaces
