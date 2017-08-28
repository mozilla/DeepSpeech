#ifndef LM_RETURN_H
#define LM_RETURN_H

#include <stdint.h>

namespace lm {
/* Structure returned by scoring routines. */
struct FullScoreReturn {
  // log10 probability
  float prob;

  /* The length of n-gram matched.  Do not use this for recombination.
   * Consider a model containing only the following n-grams:
   * -1 foo
   * -3.14  bar
   * -2.718 baz -5
   * -6 foo bar
   *
   * If you score ``bar'' then ngram_length is 1 and recombination state is the
   * empty string because bar has zero backoff and does not extend to the
   * right.
   * If you score ``foo'' then ngram_length is 1 and recombination state is
   * ``foo''.
   *
   * Ideally, keep output states around and compare them.  Failing that,
   * get out_state.ValidLength() and use that length for recombination.
   */
  unsigned char ngram_length;

  /* Left extension information.  If independent_left is set, then prob is
   * independent of words to the left (up to additional backoff).  Otherwise,
   * extend_left indicates how to efficiently extend further to the left.
   */
  bool independent_left;
  uint64_t extend_left; // Defined only if independent_left

  // Rest cost for extension to the left.
  float rest;
};

} // namespace lm
#endif // LM_RETURN_H
