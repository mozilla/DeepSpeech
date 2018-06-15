#ifndef UTIL_FLOAT_TO_STRING_H
#define UTIL_FLOAT_TO_STRING_H

// Just for ToStringBuf
#include "util/integer_to_string.hh"

namespace util {

template <> struct ToStringBuf<double> {
  // DoubleToStringConverter::kBase10MaximalLength + 1 for null paranoia.
  static const unsigned kBytes = 19;
};

// Single wasn't documented in double conversion, so be conservative and
// say the same as double.
template <> struct ToStringBuf<float> {
  static const unsigned kBytes = 19;
};

char *ToString(double value, char *to);
char *ToString(float value, char *to);

} // namespace util

#endif // UTIL_FLOAT_TO_STRING_H
