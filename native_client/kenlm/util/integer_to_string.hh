#ifndef UTIL_INTEGER_TO_STRING_H
#define UTIL_INTEGER_TO_STRING_H
#include <cstddef>
#include <stdint.h>

namespace util {

/* These functions convert integers to strings and return the end pointer.
 */
char *ToString(uint32_t value, char *to);
char *ToString(uint64_t value, char *to);

// Implemented as wrappers to above
char *ToString(int32_t value, char *to);
char *ToString(int64_t value, char *to);

// Calls the 32-bit versions for now.
char *ToString(uint16_t value, char *to);
char *ToString(int16_t value, char *to);

char *ToString(const void *value, char *to);

inline char *ToString(bool value, char *to) {
  *to++ = '0' + value;
  return to;
}

// How many bytes to reserve in the buffer for these strings:
// g++ 4.9.1 doesn't work with this:
// static const std::size_t kBytes = 5;
// So use enum.
template <class T> struct ToStringBuf;
template <> struct ToStringBuf<bool> {
  enum { kBytes = 1 };
};
template <> struct ToStringBuf<uint16_t> {
  enum { kBytes = 5 };
};
template <> struct ToStringBuf<int16_t> {
  enum { kBytes = 6 };
};
template <> struct ToStringBuf<uint32_t> {
  enum { kBytes = 10 };
};
template <> struct ToStringBuf<int32_t> {
  enum { kBytes = 11 };
};
template <> struct ToStringBuf<uint64_t> {
  enum { kBytes = 20 };
};
template <> struct ToStringBuf<int64_t> {
  // Not a typo.  2^63 has 19 digits.
  enum { kBytes = 20 };
};

template <> struct ToStringBuf<const void*> {
  // Either 18 on 64-bit or 10 on 32-bit.
  enum { kBytes = sizeof(const void*) * 2 + 2 };
};

// Maximum over this and float.
enum { kToStringMaxBytes = 20 };

} // namespace util

#endif // UTIL_INTEGER_TO_STRING_H
