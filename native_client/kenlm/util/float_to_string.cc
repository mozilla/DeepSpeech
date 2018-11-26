#include "util/float_to_string.hh"

#include "util/double-conversion/double-conversion.h"
#include "util/double-conversion/utils.h"

namespace util {
namespace {
const kenlm_double_conversion::DoubleToStringConverter kConverter(kenlm_double_conversion::DoubleToStringConverter::NO_FLAGS, "inf", "NaN", 'e', -6, 21, 6, 0);
} // namespace

char *ToString(double value, char *to) {
  kenlm_double_conversion::StringBuilder builder(to, ToStringBuf<double>::kBytes);
  kConverter.ToShortest(value, &builder);
  return &to[builder.position()];
}

char *ToString(float value, char *to) {
  kenlm_double_conversion::StringBuilder builder(to, ToStringBuf<float>::kBytes);
  kConverter.ToShortestSingle(value, &builder);
  return &to[builder.position()];
}

} // namespace util
