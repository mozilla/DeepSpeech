#include "lm/trie.hh"

#include "lm/bhiksha.hh"
#include "util/bit_packing.hh"
#include "util/exception.hh"
#include "util/sorted_uniform.hh"

#include <cassert>

namespace lm {
namespace ngram {
namespace trie {
namespace {

class KeyAccessor {
  public:
    KeyAccessor(const void *base, uint64_t key_mask, uint8_t key_bits, uint8_t total_bits)
      : base_(reinterpret_cast<const uint8_t*>(base)), key_mask_(key_mask), key_bits_(key_bits), total_bits_(total_bits) {}

    typedef uint64_t Key;

    Key operator()(uint64_t index) const {
      return util::ReadInt57(base_, index * static_cast<uint64_t>(total_bits_), key_bits_, key_mask_);
    }

  private:
    const uint8_t *const base_;
    const WordIndex key_mask_;
    const uint8_t key_bits_, total_bits_;
};

bool FindBitPacked(const void *base, uint64_t key_mask, uint8_t key_bits, uint8_t total_bits, uint64_t begin_index, uint64_t end_index, const uint64_t max_vocab, const uint64_t key, uint64_t &at_index) {
  KeyAccessor accessor(base, key_mask, key_bits, total_bits);
  if (!util::BoundedSortedUniformFind<uint64_t, KeyAccessor, util::PivotSelect<sizeof(WordIndex)>::T>(accessor, begin_index - 1, (uint64_t)0, end_index, max_vocab, key, at_index)) return false;
  return true;
}
} // namespace

uint64_t BitPacked::BaseSize(uint64_t entries, uint64_t max_vocab, uint8_t remaining_bits) {
  uint8_t total_bits = util::RequiredBits(max_vocab) + remaining_bits;
  // Extra entry for next pointer at the end.
  // +7 then / 8 to round up bits and convert to bytes
  // +sizeof(uint64_t) so that ReadInt57 etc don't go segfault.
  // Note that this waste is O(order), not O(number of ngrams).
  return ((1 + entries) * total_bits + 7) / 8 + sizeof(uint64_t);
}

void BitPacked::BaseInit(void *base, uint64_t max_vocab, uint8_t remaining_bits) {
  util::BitPackingSanity();
  word_bits_ = util::RequiredBits(max_vocab);
  word_mask_ = (1ULL << word_bits_) - 1ULL;
  if (word_bits_ > 57) UTIL_THROW(util::Exception, "Sorry, word indices more than " << (1ULL << 57) << " are not implemented.  Edit util/bit_packing.hh and fix the bit packing functions.");
  total_bits_ = word_bits_ + remaining_bits;

  base_ = static_cast<uint8_t*>(base);
  insert_index_ = 0;
  max_vocab_ = max_vocab;
}

template <class Bhiksha> uint64_t BitPackedMiddle<Bhiksha>::Size(uint8_t quant_bits, uint64_t entries, uint64_t max_vocab, uint64_t max_ptr, const Config &config) {
  return Bhiksha::Size(entries + 1, max_ptr, config) + BaseSize(entries, max_vocab, quant_bits + Bhiksha::InlineBits(entries + 1, max_ptr, config));
}

template <class Bhiksha> BitPackedMiddle<Bhiksha>::BitPackedMiddle(void *base, uint8_t quant_bits, uint64_t entries, uint64_t max_vocab, uint64_t max_next, const BitPacked &next_source, const Config &config) :
  BitPacked(),
  quant_bits_(quant_bits),
  // If the offset of the method changes, also change TrieSearch::UpdateConfigFromBinary.
  bhiksha_(base, entries + 1, max_next, config),
  next_source_(&next_source) {
  if (entries + 1 >= (1ULL << 57) || (max_next >= (1ULL << 57)))  UTIL_THROW(util::Exception, "Sorry, this does not support more than " << (1ULL << 57) << " n-grams of a particular order.  Edit util/bit_packing.hh and fix the bit packing functions.");
  BaseInit(reinterpret_cast<uint8_t*>(base) + Bhiksha::Size(entries + 1, max_next, config), max_vocab, quant_bits_ + bhiksha_.InlineBits());
}

template <class Bhiksha> util::BitAddress BitPackedMiddle<Bhiksha>::Insert(WordIndex word) {
  assert(word <= word_mask_);
  uint64_t at_pointer = insert_index_ * total_bits_;

  util::WriteInt57(base_, at_pointer, word_bits_, word);
  at_pointer += word_bits_;
  util::BitAddress ret(base_, at_pointer);
  at_pointer += quant_bits_;
  uint64_t next = next_source_->InsertIndex();
  bhiksha_.WriteNext(base_, at_pointer, insert_index_, next);
  ++insert_index_;
  return ret;
}

template <class Bhiksha> util::BitAddress BitPackedMiddle<Bhiksha>::Find(WordIndex word, NodeRange &range, uint64_t &pointer) const {
  uint64_t at_pointer;
  if (!FindBitPacked(base_, word_mask_, word_bits_, total_bits_, range.begin, range.end, max_vocab_, word, at_pointer)) {
    return util::BitAddress(NULL, 0);
  }
  pointer = at_pointer;
  at_pointer *= total_bits_;
  at_pointer += word_bits_;
  bhiksha_.ReadNext(base_, at_pointer + quant_bits_, pointer, total_bits_, range);

  return util::BitAddress(base_, at_pointer);
}

template <class Bhiksha> void BitPackedMiddle<Bhiksha>::FinishedLoading(uint64_t next_end, const Config &config) {
  // Write at insert_index. . .
  uint64_t last_next_write = insert_index_ * total_bits_ +
    // at the offset where the next pointers are stored.
    (total_bits_ - bhiksha_.InlineBits());
  bhiksha_.WriteNext(base_, last_next_write, insert_index_, next_end);
  bhiksha_.FinishedLoading(config);
}

util::BitAddress BitPackedLongest::Insert(WordIndex index) {
  assert(index <= word_mask_);
  uint64_t at_pointer = insert_index_ * total_bits_;
  util::WriteInt57(base_, at_pointer, word_bits_, index);
  at_pointer += word_bits_;
  ++insert_index_;
  return util::BitAddress(base_, at_pointer);
}

util::BitAddress BitPackedLongest::Find(WordIndex word, const NodeRange &range) const {
  uint64_t at_pointer;
  if (!FindBitPacked(base_, word_mask_, word_bits_, total_bits_, range.begin, range.end, max_vocab_, word, at_pointer)) return util::BitAddress(NULL, 0);
  at_pointer = at_pointer * total_bits_ + word_bits_;
  return util::BitAddress(base_, at_pointer);
}

template class BitPackedMiddle<DontBhiksha>;
template class BitPackedMiddle<ArrayBhiksha>;

} // namespace trie
} // namespace ngram
} // namespace lm
