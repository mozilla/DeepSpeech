/* Downloaded from http://sites.google.com/site/murmurhash/ which says "All
 * code is released to the public domain. For business purposes, Murmurhash is
 * under the MIT license."
 * This is modified from the original:
 * ULL tag on 0xc6a4a7935bd1e995 so this will compile on 32-bit.
 * length changed to unsigned int.
 * placed in namespace util
 * add MurmurHashNative
 * default option = 0 for seed
 * ARM port from NICT
 */

#include "util/murmur_hash.hh"
#include <cstring>

namespace util {

//-----------------------------------------------------------------------------
// MurmurHash2, 64-bit versions, by Austin Appleby

// The same caveats as 32-bit MurmurHash2 apply here - beware of alignment
// and endian-ness issues if used across multiple platforms.

// 64-bit hash for 64-bit platforms

uint64_t MurmurHash64A ( const void * key, std::size_t len, uint64_t seed )
{
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

#if defined(__arm) || defined(__arm__)
  const size_t ksize = sizeof(uint64_t);
  const unsigned char * data = (const unsigned char *)key;
  const unsigned char * end = data + (std::size_t)(len/8) * ksize;
#else
  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);
#endif

  while(data != end)
  {
#if defined(__arm) || defined(__arm__)
    uint64_t k;
    memcpy(&k, data, ksize);
    data += ksize;
#else
    uint64_t k = *data++;
#endif

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const unsigned char * data2 = (const unsigned char*)data;

  switch(len & 7)
  {
  case 7: h ^= uint64_t(data2[6]) << 48;
  case 6: h ^= uint64_t(data2[5]) << 40;
  case 5: h ^= uint64_t(data2[4]) << 32;
  case 4: h ^= uint64_t(data2[3]) << 24;
  case 3: h ^= uint64_t(data2[2]) << 16;
  case 2: h ^= uint64_t(data2[1]) << 8;
  case 1: h ^= uint64_t(data2[0]);
          h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}


// 64-bit hash for 32-bit platforms

uint64_t MurmurHash64B ( const void * key, std::size_t len, uint64_t seed )
{
  const unsigned int m = 0x5bd1e995;
  const int r = 24;

  unsigned int h1 = seed ^ len;
  unsigned int h2 = 0;

#if defined(__arm) || defined(__arm__)
  size_t ksize = sizeof(unsigned int);
  const unsigned char * data = (const unsigned char *)key;
#else
  const unsigned int * data = (const unsigned int *)key;
#endif

  unsigned int k1, k2;
  while(len >= 8)
  {
#if defined(__arm) || defined(__arm__)
    memcpy(&k1, data, ksize);
    data += ksize;
    memcpy(&k2, data, ksize);
    data += ksize;
#else
    k1 = *data++;
    k2 = *data++;
#endif

    k1 *= m; k1 ^= k1 >> r; k1 *= m;
    h1 *= m; h1 ^= k1;
    len -= 4;

    k2 *= m; k2 ^= k2 >> r; k2 *= m;
    h2 *= m; h2 ^= k2;
    len -= 4;
  }

  if(len >= 4)
  {
#if defined(__arm) || defined(__arm__)
    memcpy(&k1, data, ksize);
    data += ksize;
#else
    k1 = *data++;
#endif
    k1 *= m; k1 ^= k1 >> r; k1 *= m;
    h1 *= m; h1 ^= k1;
    len -= 4;
  }

  switch(len)
  {
  case 3: h2 ^= ((unsigned char*)data)[2] << 16;
  case 2: h2 ^= ((unsigned char*)data)[1] << 8;
  case 1: h2 ^= ((unsigned char*)data)[0];
      h2 *= m;
  };

  h1 ^= h2 >> 18; h1 *= m;
  h2 ^= h1 >> 22; h2 *= m;
  h1 ^= h2 >> 17; h1 *= m;
  h2 ^= h1 >> 19; h2 *= m;

  uint64_t h = h1;

  h = (h << 32) | h2;

  return h;
}

// Trick to test for 64-bit architecture at compile time.
namespace {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
template <unsigned L> inline uint64_t MurmurHashNativeBackend(const void * key, std::size_t len, uint64_t seed) {
  return MurmurHash64A(key, len, seed);
}
template <> inline uint64_t MurmurHashNativeBackend<4>(const void * key, std::size_t len, uint64_t seed) {
  return MurmurHash64B(key, len, seed);
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace

uint64_t MurmurHashNative(const void * key, std::size_t len, uint64_t seed) {
  return MurmurHashNativeBackend<sizeof(void*)>(key, len, seed);
}

} // namespace util
