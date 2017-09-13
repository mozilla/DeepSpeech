#ifndef UTIL_MURMUR_HASH_H
#define UTIL_MURMUR_HASH_H
#include <cstddef>
#include <stdint.h>

namespace util {

// 64-bit machine version
uint64_t MurmurHash64A(const void * key, std::size_t len, uint64_t seed = 0);
// 32-bit machine version (not the same function as above)
uint64_t MurmurHash64B(const void * key, std::size_t len, uint64_t seed = 0);
// Use the version for this arch.  Because the values differ across
// architectures, really only use it for in-memory structures.
uint64_t MurmurHashNative(const void * key, std::size_t len, uint64_t seed = 0);

} // namespace util

#endif // UTIL_MURMUR_HASH_H
