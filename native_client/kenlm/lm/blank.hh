#ifndef LM_BLANK_H
#define LM_BLANK_H

#include <limits>
#include <stdint.h>
#include <cmath>

namespace lm {
namespace ngram {

/* Suppose "foo bar" appears with zero backoff but there is no trigram
 * beginning with these words.  Then, when scoring "foo bar", the model could
 * return out_state containing "bar" or even null context if "bar" also has no
 * backoff and is never followed by another word.  Then the backoff is set to
 * kNoExtensionBackoff.  If the n-gram might be extended, then out_state must
 * contain the full n-gram, in which case kExtensionBackoff is set.  In any
 * case, if an n-gram has non-zero backoff, the full state is returned so
 * backoff can be properly charged.
 * These differ only in sign bit because the backoff is in fact zero in either
 * case.
 */
const float kNoExtensionBackoff = -0.0;
const float kExtensionBackoff = 0.0;
const uint64_t kNoExtensionQuant = 0;
const uint64_t kExtensionQuant = 1;

inline void SetExtension(float &backoff) {
  if (backoff == kNoExtensionBackoff) backoff = kExtensionBackoff;
}

// This compiles down nicely.
inline bool HasExtension(const float &backoff) {
  typedef union { float f; uint32_t i; } UnionValue;
  UnionValue compare, interpret;
  compare.f = kNoExtensionBackoff;
  interpret.f = backoff;
  return compare.i != interpret.i;
}

} // namespace ngram
} // namespace lm
#endif // LM_BLANK_H
