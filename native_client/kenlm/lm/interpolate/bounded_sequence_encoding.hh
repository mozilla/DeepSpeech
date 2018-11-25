#ifndef LM_INTERPOLATE_BOUNDED_SEQUENCE_ENCODING_H
#define LM_INTERPOLATE_BOUNDED_SEQUENCE_ENCODING_H

/* Encodes fixed-length sequences of integers with known bounds on each entry.
 * This is used to encode how far each model has backed off.
 * TODO: make this class efficient.  Bit-level packing or multiply by bound and
 * add.
 */

#include "util/exception.hh"
#include "util/fixed_array.hh"

#if BYTE_ORDER != LITTLE_ENDIAN
#warning The interpolation code assumes little endian for now.
#endif

#include <algorithm>
#include <cstring>

namespace lm {
namespace interpolate {

class BoundedSequenceEncoding {
  public:
    // Encode [0, bound_begin[0]) x [0, bound_begin[1]) x [0, bound_begin[2]) x ... x [0, *(bound_end - 1)) for entries in the sequence
    BoundedSequenceEncoding(const unsigned char *bound_begin, const unsigned char *bound_end);

    std::size_t Entries() const { return entries_.size(); }

    std::size_t EncodedLength() const { return byte_length_; }

    void Encode(const unsigned char *from, void *to_void) const {
      uint8_t *to = static_cast<uint8_t*>(to_void);
      uint64_t cur = 0;
      for (const Entry *i = entries_.begin(); i != entries_.end(); ++i, ++from) {
        if (UTIL_UNLIKELY(i->next)) {
          std::memcpy(to, &cur, sizeof(uint64_t));
          to += sizeof(uint64_t);
          cur = 0;
        }
        cur |= static_cast<uint64_t>(*from) << i->shift;
      }
      memcpy(to, &cur, overhang_);
    }

    void Decode(const void *from_void, unsigned char *to) const {
      const uint8_t *from = static_cast<const uint8_t*>(from_void);
      uint64_t cur = 0;
      memcpy(&cur, from, first_copy_);
      for (const Entry *i = entries_.begin(); i != entries_.end(); ++i, ++to) {
        if (UTIL_UNLIKELY(i->next)) {
          from += sizeof(uint64_t);
          cur = 0;
          std::memcpy(&cur, from,
              std::min<std::size_t>(sizeof(uint64_t), static_cast<const uint8_t*>(from_void) + byte_length_ - from));
        }
        *to = (cur >> i->shift) & i->mask;
      }
    }

  private:
    struct Entry {
      bool next;
      uint8_t shift;
      uint64_t mask;
    };
    util::FixedArray<Entry> entries_;
    std::size_t byte_length_;
    std::size_t first_copy_;
    std::size_t overhang_;
};


}} // namespaces

#endif // LM_INTERPOLATE_BOUNDED_SEQUENCE_ENCODING_H
