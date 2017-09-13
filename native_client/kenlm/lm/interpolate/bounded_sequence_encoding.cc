#include "lm/interpolate/bounded_sequence_encoding.hh"

#include <algorithm>

namespace lm { namespace interpolate {

BoundedSequenceEncoding::BoundedSequenceEncoding(const unsigned char *bound_begin, const unsigned char *bound_end)
  : entries_(bound_end - bound_begin) {
  std::size_t full = 0;
  Entry entry;
  entry.shift = 0;
  for (const unsigned char *i = bound_begin; i != bound_end; ++i) {
    uint8_t length;
    if (*i <= 1) {
      length = 0;
    } else {
      length = sizeof(unsigned int) * 8 - __builtin_clz((unsigned int)*i);
    }
    entry.mask = (1ULL << length) - 1ULL;
    if (entry.shift + length > 64) {
      entry.shift = 0;
      entry.next = true;
      ++full;
    } else {
      entry.next = false;
    }
    entries_.push_back(entry);
    entry.shift += length;
  }
  byte_length_ = full * sizeof(uint64_t) + (entry.shift + 7) / 8;
  first_copy_ = std::min<std::size_t>(byte_length_, sizeof(uint64_t));
  // Size of last uint64_t.  Zero if empty, otherwise [1,8] depending on mod.
  overhang_ = byte_length_ == 0 ? 0 : ((byte_length_ - 1) % 8 + 1);
}

}} // namespaces
