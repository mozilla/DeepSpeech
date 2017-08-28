#ifndef LM_STATE_H
#define LM_STATE_H

#include "lm/max_order.hh"
#include "lm/word_index.hh"
#include "util/murmur_hash.hh"

#include <cstring>

namespace lm {
namespace ngram {

// This is a POD but if you want memcmp to return the same as operator==, call
// ZeroRemaining first.
class State {
  public:
    bool operator==(const State &other) const {
      if (length != other.length) return false;
      return !memcmp(words, other.words, length * sizeof(WordIndex));
    }

    // Three way comparison function.
    int Compare(const State &other) const {
      if (length != other.length) return length < other.length ? -1 : 1;
      return memcmp(words, other.words, length * sizeof(WordIndex));
    }

    bool operator<(const State &other) const {
      if (length != other.length) return length < other.length;
      return memcmp(words, other.words, length * sizeof(WordIndex)) < 0;
    }

    // Call this before using raw memcmp.
    void ZeroRemaining() {
      for (unsigned char i = length; i < KENLM_MAX_ORDER - 1; ++i) {
        words[i] = 0;
        backoff[i] = 0.0;
      }
    }

    unsigned char Length() const { return length; }

    // You shouldn't need to touch anything below this line, but the members are public so FullState will qualify as a POD.
    // This order minimizes total size of the struct if WordIndex is 64 bit, float is 32 bit, and alignment of 64 bit integers is 64 bit.
    WordIndex words[KENLM_MAX_ORDER - 1];
    float backoff[KENLM_MAX_ORDER - 1];
    unsigned char length;
};

typedef State Right;

inline uint64_t hash_value(const State &state, uint64_t seed = 0) {
  return util::MurmurHashNative(state.words, sizeof(WordIndex) * state.length, seed);
}

struct Left {
  bool operator==(const Left &other) const {
    return
      length == other.length &&
      (!length || (pointers[length - 1] == other.pointers[length - 1] && full == other.full));
  }

  int Compare(const Left &other) const {
    if (length < other.length) return -1;
    if (length > other.length) return 1;
    if (length == 0) return 0; // Must be full.
    if (pointers[length - 1] > other.pointers[length - 1]) return 1;
    if (pointers[length - 1] < other.pointers[length - 1]) return -1;
    return (int)full - (int)other.full;
  }

  bool operator<(const Left &other) const {
    return Compare(other) == -1;
  }

  void ZeroRemaining() {
    for (uint64_t * i = pointers + length; i < pointers + KENLM_MAX_ORDER - 1; ++i)
      *i = 0;
  }

  uint64_t pointers[KENLM_MAX_ORDER - 1];
  unsigned char length;
  bool full;
};

inline uint64_t hash_value(const Left &left) {
  unsigned char add[2];
  add[0] = left.length;
  add[1] = left.full;
  return util::MurmurHashNative(add, 2, left.length ? left.pointers[left.length - 1] : 0);
}

struct ChartState {
  bool operator==(const ChartState &other) const {
    return (right == other.right) && (left == other.left);
  }

  int Compare(const ChartState &other) const {
    int lres = left.Compare(other.left);
    if (lres) return lres;
    return right.Compare(other.right);
  }

  bool operator<(const ChartState &other) const {
    return Compare(other) < 0;
  }

  void ZeroRemaining() {
    left.ZeroRemaining();
    right.ZeroRemaining();
  }

  Left left;
  State right;
};

inline uint64_t hash_value(const ChartState &state) {
  return hash_value(state.right, hash_value(state.left));
}


} // namespace ngram
} // namespace lm

#endif // LM_STATE_H
