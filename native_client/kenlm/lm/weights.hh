#ifndef LM_WEIGHTS_H
#define LM_WEIGHTS_H

// Weights for n-grams.  Probability and possibly a backoff.

namespace lm {
struct Prob {
  float prob;
};
// No inheritance so this will be a POD.
struct ProbBackoff {
  float prob;
  float backoff;
};
struct RestWeights {
  float prob;
  float backoff;
  float rest;
};

} // namespace lm
#endif // LM_WEIGHTS_H
