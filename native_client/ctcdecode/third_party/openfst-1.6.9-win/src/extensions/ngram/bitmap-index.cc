// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/ngram/bitmap-index.h>

#include <algorithm>
#include <iterator>

#include <fst/log.h>
#include <fst/extensions/ngram/nthbit.h>

namespace fst {
namespace {
const size_t kPrimaryBlockBits =
    BitmapIndex::kStorageBitSize * BitmapIndex::kSecondaryBlockSize;

// If [begin, begin+size) is a monotonically increasing running sum of
// popcounts for a bitmap, this will return the index of the word that contains
// the value'th zero.  If value is larger then the number of zeros in the
// bitmap, size will be returned.  The idea is that the number of zerocounts
// (i.e. the popcount of logical NOT of values) is offset * kStorageBitSize
// minus the value for each element of the running sum.
template <size_t BlockSize, typename Container>
size_t InvertedSearch(const Container& c,
                      size_t first_idx,
                      size_t last_idx,
                      size_t value) {
  const size_t begin_idx = first_idx;
  while (first_idx != last_idx) {
    // Invariant: [first_idx, last_idx) is the search range.
    size_t mid_idx = first_idx + ((last_idx - first_idx) / 2);
    size_t mid_value = BlockSize * (1 + (mid_idx - begin_idx)) - c[mid_idx];
    if (mid_value < value) {
      first_idx = mid_idx + 1;
    } else {
      last_idx = mid_idx;
    }
  }
  return first_idx;
}
}  // namespace

size_t BitmapIndex::Rank1(size_t end) const {
  if (end == 0) return 0;
  const uint32 end_word = (end - 1) >> BitmapIndex::kStorageLogBitSize;
  const uint32 sum = get_index_ones_count(end_word);
  const size_t masked = end & kStorageBlockMask;
  if (masked == 0) {
    return sum + __builtin_popcountll(bits_[end_word]);
  } else {
    const uint64 zero = 0;
    return sum + __builtin_popcountll(bits_[end_word] &
                                      (~zero >> (kStorageBitSize - masked)));
  }
}

size_t BitmapIndex::Select1(size_t bit_index) const {
  if (bit_index >= GetOnesCount()) return Bits();
  // search primary index for the relevant block
  uint32 rembits = bit_index + 1;
  const uint32 block = find_primary_block(bit_index + 1);
  uint32 offset = 0;
  if (block > 0) {
    rembits -= primary_index_[block - 1];
    offset += block * kSecondaryBlockSize;
  }
  // search the secondary index
  uint32 word = find_secondary_block(offset, rembits);
  if (word > 0) {
    rembits -= secondary_index_[offset + word - 1];
    offset += word;
  }
  int nth = nth_bit(bits_[offset], rembits);
  return (offset << BitmapIndex::kStorageLogBitSize) + nth;
}

size_t BitmapIndex::Select0(size_t bit_index) const {
  if (bit_index >= Bits() - GetOnesCount()) return Bits();
  // search inverted primary index for relevant block
  uint32 remzeros = bit_index + 1;
  uint32 offset = 0;
  const uint32 block = find_inverted_primary_block(bit_index + 1);
  if (block > 0) {
    remzeros -= kPrimaryBlockBits * block - primary_index_[block - 1];
    offset += block * kSecondaryBlockSize;
  }
  // search the inverted secondary index
  uint32 word = find_inverted_secondary_block(offset, remzeros);
  if (word > 0) {
    remzeros -= BitmapIndex::kStorageBitSize * word -
                secondary_index_[offset + word - 1];
    offset += word;
  }
  int nth = nth_bit(~bits_[offset], remzeros);
  return (offset << BitmapIndex::kStorageLogBitSize) + nth;
}

std::pair<size_t, size_t> BitmapIndex::Select0s(size_t bit_index) const {
  const uint64 zero = 0;
  const uint64 ones = ~zero;
  size_t zeros_count = Bits() - GetOnesCount();
  if (bit_index >= zeros_count) return std::make_pair(Bits(), Bits());
  if (bit_index + 1 >= zeros_count) {
    return std::make_pair(Select0(bit_index), Bits());
  }
  // search inverted primary index for relevant block
  uint32 remzeros = bit_index + 1;
  uint32 offset = 0;
  const uint32 block = find_inverted_primary_block(bit_index + 1);
  size_t num_zeros_in_block =
      kPrimaryBlockBits * (1 + block) - primary_index_[block];
  if (block > 0) {
    size_t num_zeros_next =
        kPrimaryBlockBits * block - primary_index_[block - 1];
    num_zeros_in_block -= num_zeros_next;
    remzeros -= num_zeros_next;
    offset += block * kSecondaryBlockSize;
  }
  // search the inverted secondary index
  uint32 word = find_inverted_secondary_block(offset, remzeros);
  uint32 sum_zeros_next_word = BitmapIndex::kStorageBitSize * (1 + word) -
                               secondary_index_[offset + word];
  uint32 sum_zeros_this_word = 0;
  if (word > 0) {
    sum_zeros_this_word = BitmapIndex::kStorageBitSize * word -
                          secondary_index_[offset + word - 1];
    remzeros -= sum_zeros_this_word;
    offset += word;
  }
  int nth = nth_bit(~bits_[offset], remzeros);
  size_t current_zero = (offset << BitmapIndex::kStorageLogBitSize) + nth;

  size_t next_zero;
  // Does the current block contain the next zero?
  if (num_zeros_in_block > remzeros + 1) {
    if (sum_zeros_next_word - sum_zeros_this_word >= remzeros + 1) {
      // the next zero is in this word
      next_zero = (offset << BitmapIndex::kStorageLogBitSize) +
                  nth_bit(~bits_[offset], remzeros + 1);
    } else {
      // Find the first field that is not all ones by linear scan.
      // In the worst case, this may scan 8Kbytes.  The alternative is
      // to inspect secondary_index_ looking for a place to jump to, but
      // that would probably use more cache.
      while (bits_[++offset] == ones) {
      }
      next_zero = (offset << BitmapIndex::kStorageLogBitSize) +
                  __builtin_ctzll(~bits_[offset]);
    }
  } else {
    // the next zero is in a different block, a full search is required.
    next_zero = Select0(bit_index + 1);
  }
  return std::make_pair(current_zero, next_zero);
}

size_t BitmapIndex::get_index_ones_count(size_t array_index) const {
  uint32 sum = 0;
  if (array_index > 0) {
    sum += secondary_index_[array_index - 1];
    uint32 end_block = (array_index - 1) / kSecondaryBlockSize;
    if (end_block > 0) sum += primary_index_[end_block - 1];
  }
  return sum;
}

void BitmapIndex::BuildIndex(const uint64 *bits, size_t size) {
  bits_ = bits;
  size_ = size;
  primary_index_.resize(primary_index_size());
  secondary_index_.resize(ArraySize());
  const uint64 zero = 0;
  const uint64 ones = ~zero;
  uint32 popcount = 0;
  for (uint32 block = 0; block * kSecondaryBlockSize < ArraySize(); block++) {
    uint32 block_popcount = 0;
    uint32 block_begin = block * kSecondaryBlockSize;
    uint32 block_end = block_begin + kSecondaryBlockSize;
    if (block_end > ArraySize()) block_end = ArraySize();
    for (uint32 j = block_begin; j < block_end; ++j) {
      uint64 mask = ones;
      if (j == ArraySize() - 1) {
        mask = ones >> (-size_ & BitmapIndex::kStorageBlockMask);
      }
      block_popcount += __builtin_popcountll(bits_[j] & mask);
      secondary_index_[j] = block_popcount;
    }
    popcount += block_popcount;
    primary_index_[block] = popcount;
  }
}

size_t BitmapIndex::find_secondary_block(size_t block_begin,
                                         size_t rem_bit_index) const {
  size_t block_end = block_begin + kSecondaryBlockSize;
  if (block_end > ArraySize()) block_end = ArraySize();
  return std::distance(
      secondary_index_.begin() + block_begin,
      std::lower_bound(secondary_index_.begin() + block_begin,
                       secondary_index_.begin() + block_end, rem_bit_index));
}

size_t BitmapIndex::find_inverted_secondary_block(size_t block_begin,
                                                  size_t rem_bit_index) const {
  size_t block_end = block_begin + kSecondaryBlockSize;
  if (block_end > ArraySize()) block_end = ArraySize();
  return InvertedSearch<BitmapIndex::kStorageBitSize>(secondary_index_,
                                                      block_begin, block_end,
                                                      rem_bit_index)
      - block_begin;
}

inline size_t BitmapIndex::find_primary_block(size_t bit_index) const {
  return std::distance(
      primary_index_.begin(),
      std::lower_bound(primary_index_.begin(),
                       primary_index_.begin() + primary_index_size(),
                       bit_index));
}

size_t BitmapIndex::find_inverted_primary_block(size_t bit_index) const {
  return InvertedSearch<kPrimaryBlockBits>(
      primary_index_, 0, primary_index_.size(), bit_index);
}
}  // end namespace fst
