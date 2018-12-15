// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_NGRAM_BITMAP_INDEX_H_
#define FST_EXTENSIONS_NGRAM_BITMAP_INDEX_H_

#include <utility>
#include <vector>

#include <fst/compat.h>

// This class is a bitstring storage class with an index that allows
// seeking to the Nth set or clear bit in time O(Log(N)) where N is
// the length of the bit vector.  In addition, it allows counting set or
// clear bits over ranges in constant time.
//
// This is accomplished by maintaining an "secondary" index of limited
// size in bits that maintains a running count of the number of bits set
// in each block of bitmap data.  A block is defined as the number of
// uint64_t values that can fit in the secondary index before an overflow
// occurs.
//
// To handle overflows, a "primary" index containing a running count of
// bits set in each block is created using the type uint64_t.

namespace fst {

class BitmapIndex {
 public:
  static size_t StorageSize(size_t size) {
    return ((size + kStorageBlockMask) >> kStorageLogBitSize);
  }

  BitmapIndex() : bits_(nullptr), size_(0) {}

  bool Get(size_t index) const {
    return (bits_[index >> kStorageLogBitSize] &
            (kOne << (index & kStorageBlockMask))) != 0;
  }

  static void Set(uint64_t* bits, size_t index) {
    bits[index >> kStorageLogBitSize] |= (kOne << (index & kStorageBlockMask));
  }

  static void Clear(uint64_t* bits, size_t index) {
    bits[index >> kStorageLogBitSize] &= ~(kOne << (index & kStorageBlockMask));
  }

  size_t Bits() const { return size_; }

  size_t ArraySize() const { return StorageSize(size_); }

  // Returns the number of one bits in the bitmap
  size_t GetOnesCount() const {
    return primary_index_[primary_index_size() - 1];
  }

  // Returns the number of one bits in positions 0 to limit - 1.
  // REQUIRES: limit <= Bits()
  size_t Rank1(size_t end) const;

  // Returns the number of one bits in the range start to end - 1.
  // REQUIRES: limit <= Bits()
  size_t GetOnesCountInRange(size_t start, size_t end) const {
    return Rank1(end) - Rank1(start);
  }

  // Returns the number of zero bits in positions 0 to limit - 1.
  // REQUIRES: limit <= Bits()
  size_t Rank0(size_t end) const { return end - Rank1(end); }

  // Returns the number of zero bits in the range start to end - 1.
  // REQUIRES: limit <= Bits()
  size_t GetZeroesCountInRange(size_t start, size_t end) const {
    return end - start - GetOnesCountInRange(start, end);
  }

  // Return true if any bit between begin inclusive and end exclusive
  // is set.  0 <= begin <= end <= Bits() is required.
  //
  bool TestRange(size_t start, size_t end) const {
    return Rank1(end) > Rank1(start);
  }

  // Returns the offset to the nth set bit (zero based)
  // or Bits() if index >= number of ones
  size_t Select1(size_t bit_index) const;

  // Returns the offset to the nth clear bit (zero based)
  // or Bits() if index > number of
  size_t Select0(size_t bit_index) const;

  // Returns the offset of the nth and nth+1 clear bit (zero based),
  // equivalent to two calls to Select0, but more efficient.
  std::pair<size_t, size_t> Select0s(size_t bit_index) const;

  // Rebuilds from index for the associated Bitmap, should be called
  // whenever changes have been made to the Bitmap or else behavior
  // of the indexed bitmap methods will be undefined.
  void BuildIndex(const uint64_t* bits, size_t size);

  // the secondary index accumulates counts until it can possibly overflow
  // this constant computes the number of uint64_t units that can fit into
  // units the size of uint16_t.
  static const uint64_t kOne = 1;
  static const uint32_t kStorageBitSize = 64;
  static const uint32_t kStorageLogBitSize = 6;
  static const uint32_t kSecondaryBlockSize =
      ((1 << 16) - 1) >> kStorageLogBitSize;

 private:
  static const uint32_t kStorageBlockMask = kStorageBitSize - 1;

  // returns, from the index, the count of ones up to array_index
  size_t get_index_ones_count(size_t array_index) const;

  // because the indexes, both primary and secondary, contain a running
  // count of the population of one bits contained in [0,i), there is
  // no reason to have an element in the zeroth position as this value would
  // necessarily be zero.  (The bits are indexed in a zero based way.)  Thus
  // we don't store the 0th element in either index.  Both of the following
  // functions, if greater than 0, must be decremented by one before retreiving
  // the value from the corresponding array.
  // returns the 1 + the block that contains the bitindex in question
  // the inverted version works the same but looks for zeros using an inverted
  // view of the index
  size_t find_primary_block(size_t bit_index) const;

  size_t find_inverted_primary_block(size_t bit_index) const;

  // similarly, the secondary index (which resets its count to zero at
  // the end of every kSecondaryBlockSize entries) does not store the element
  // at 0.  Note that the rem_bit_index parameter is the number of bits
  // within the secondary block, after the bits accounted for by the primary
  // block have been removed (i.e. the remaining bits)  And, because we
  // reset to zero with each new block, there is no need to store those
  // actual zeros.
  // returns 1 + the secondary block that contains the bitindex in question
  size_t find_secondary_block(size_t block, size_t rem_bit_index) const;

  size_t find_inverted_secondary_block(size_t block,
                                       size_t rem_bit_index) const;

  // We create a primary index based upon the number of secondary index
  // blocks.  The primary index uses fields wide enough to accomodate any
  // index of the bitarray so cannot overflow
  // The primary index is the actual running
  // count of one bits set for all blocks (and, thus, all uint64_ts).
  size_t primary_index_size() const {
    return (ArraySize() + kSecondaryBlockSize - 1) / kSecondaryBlockSize;
  }

  const uint64_t* bits_;
  size_t size_;

  // The primary index contains the running popcount of all blocks
  // which means the nth value contains the popcounts of
  // [0,n*kSecondaryBlockSize], however, the 0th element is omitted.
  std::vector<uint32_t> primary_index_;
  // The secondary index contains the running popcount of the associated
  // bitmap.  It is the same length (in units of uint16_t) as the
  // bitmap's map is in units of uint64_ts.
  std::vector<uint16_t> secondary_index_;
};

}  // end namespace fst

#endif  // FST_EXTENSIONS_NGRAM_BITMAP_INDEX_H_
