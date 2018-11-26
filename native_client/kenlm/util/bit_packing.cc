#include "util/bit_packing.hh"
#include "util/exception.hh"

#include <cstring>

namespace util {

namespace {
template <bool> struct StaticCheck {};
template <> struct StaticCheck<true> { typedef bool StaticAssertionPassed; };

// If your float isn't 4 bytes, we're hosed.
typedef StaticCheck<sizeof(float) == 4>::StaticAssertionPassed FloatSize;

} // namespace

uint8_t RequiredBits(uint64_t max_value) {
  if (!max_value) return 0;
  uint8_t ret = 1;
  while (max_value >>= 1) ++ret;
  return ret;
}

void BitPackingSanity() {
  const FloatEnc neg1 = { -1.0 }, pos1 = { 1.0 };
  if ((neg1.i ^ pos1.i) != 0x80000000) UTIL_THROW(Exception, "Sign bit is not 0x80000000");
  char mem[57+8];
  memset(mem, 0, sizeof(mem));
  const uint64_t test57 = 0x123456789abcdefULL;
  for (uint64_t b = 0; b < 57 * 8; b += 57) {
    WriteInt57(mem, b, 57, test57);
  }
  for (uint64_t b = 0; b < 57 * 8; b += 57) {
    if (test57 != ReadInt57(mem, b, 57, (1ULL << 57) - 1))
      UTIL_THROW(Exception, "The bit packing routines are failing for your architecture.  Please send a bug report with your architecture, operating system, and compiler.");
  }
  // TODO: more checks.
}

} // namespace util
