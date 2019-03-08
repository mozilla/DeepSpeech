// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_NGRAM_NTHBIT_H_
#define FST_EXTENSIONS_NGRAM_NTHBIT_H_

#include <fst/types.h>
#include <fst/compat.h>

extern uint32_t nth_bit_bit_offset[];

inline uint32_t nth_bit(uint64_t v, uint32_t r) {
  uint32_t shift = 0;
  uint32_t c = __builtin_popcount(v & 0xffffffff);
  uint32_t mask = -(r > c);
  r -= c & mask;
  shift += (32 & mask);

  c = __builtin_popcount((v >> shift) & 0xffff);
  mask = -(r > c);
  r -= c & mask;
  shift += (16 & mask);

  c = __builtin_popcount((v >> shift) & 0xff);
  mask = -(r > c);
  r -= c & mask;
  shift += (8 & mask);

  return shift +
         ((nth_bit_bit_offset[(v >> shift) & 0xff] >> ((r - 1) << 2)) & 0xf);
}

#endif  // FST_EXTENSIONS_NGRAM_NTHBIT_H_
