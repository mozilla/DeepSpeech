#ifndef LM_COMMON_COMPARE_H
#define LM_COMMON_COMPARE_H

#include "lm/common/ngram.hh"
#include "lm/word_index.hh"

#include <functional>
#include <string>

namespace lm {

/**
 * Abstract parent class for defining custom n-gram comparators.
 */
template <class Child> class Comparator : public std::binary_function<const void *, const void *, bool> {
  public:

    /**
     * Constructs a comparator capable of comparing two n-grams.
     *
     * @param order Number of words in each n-gram
     */
    explicit Comparator(std::size_t order) : order_(order) {}

    /**
     * Applies the comparator using the Compare method that must be defined in any class that inherits from this class.
     *
     * @param lhs A pointer to the n-gram on the left-hand side of the comparison
     * @param rhs A pointer to the n-gram on the right-hand side of the comparison
     *
     * @see ContextOrder::Compare
     * @see PrefixOrder::Compare
     * @see SuffixOrder::Compare
     */
    inline bool operator()(const void *lhs, const void *rhs) const {
      return static_cast<const Child*>(this)->Compare(static_cast<const WordIndex*>(lhs), static_cast<const WordIndex*>(rhs));
    }

    /** Gets the n-gram order defined for this comparator. */
    std::size_t Order() const { return order_; }

  protected:
    std::size_t order_;
};

/**
 * N-gram comparator that compares n-grams according to their reverse (suffix) order.
 *
 * This comparator compares n-grams lexicographically, one word at a time,
 * beginning with the last word of each n-gram and ending with the first word of each n-gram.
 *
 * Some examples of n-gram comparisons as defined by this comparator:
 * - a b c == a b c
 * - a b c < a b d
 * - a b c > a d b
 * - a b c > a b b
 * - a b c > x a c
 * - a b c < x y z
 */
class SuffixOrder : public Comparator<SuffixOrder> {
  public:

    /**
     * Constructs a comparator capable of comparing two n-grams.
     *
     * @param order Number of words in each n-gram
     */
    explicit SuffixOrder(std::size_t order) : Comparator<SuffixOrder>(order) {}

    /**
     * Compares two n-grams lexicographically, one word at a time,
     * beginning with the last word of each n-gram and ending with the first word of each n-gram.
     *
     * @param lhs A pointer to the n-gram on the left-hand side of the comparison
     * @param rhs A pointer to the n-gram on the right-hand side of the comparison
     */
    inline bool Compare(const WordIndex *lhs, const WordIndex *rhs) const {
      for (std::size_t i = order_ - 1; i != 0; --i) {
        if (lhs[i] != rhs[i])
          return lhs[i] < rhs[i];
      }
      return lhs[0] < rhs[0];
    }

    static const unsigned kMatchOffset = 1;
};


/**
  * N-gram comparator that compares n-grams according to the reverse (suffix) order of the n-gram context.
  *
  * This comparator compares n-grams lexicographically, one word at a time,
  * beginning with the penultimate word of each n-gram and ending with the first word of each n-gram;
  * finally, this comparator compares the last word of each n-gram.
  *
  * Some examples of n-gram comparisons as defined by this comparator:
  * - a b c == a b c
  * - a b c < a b d
  * - a b c < a d b
  * - a b c > a b b
  * - a b c > x a c
  * - a b c < x y z
  */
class ContextOrder : public Comparator<ContextOrder> {
  public:

    /**
     * Constructs a comparator capable of comparing two n-grams.
     *
     * @param order Number of words in each n-gram
     */
    explicit ContextOrder(std::size_t order) : Comparator<ContextOrder>(order) {}

    /**
     * Compares two n-grams lexicographically, one word at a time,
     * beginning with the penultimate word of each n-gram and ending with the first word of each n-gram;
     * finally, this comparator compares the last word of each n-gram.
     *
     * @param lhs A pointer to the n-gram on the left-hand side of the comparison
     * @param rhs A pointer to the n-gram on the right-hand side of the comparison
     */
    inline bool Compare(const WordIndex *lhs, const WordIndex *rhs) const {
      for (int i = order_ - 2; i >= 0; --i) {
        if (lhs[i] != rhs[i])
          return lhs[i] < rhs[i];
      }
      return lhs[order_ - 1] < rhs[order_ - 1];
    }
};

/**
 * N-gram comparator that compares n-grams according to their natural (prefix) order.
 *
 * This comparator compares n-grams lexicographically, one word at a time,
 * beginning with the first word of each n-gram and ending with the last word of each n-gram.
 *
 * Some examples of n-gram comparisons as defined by this comparator:
 * - a b c == a b c
 * - a b c < a b d
 * - a b c < a d b
 * - a b c > a b b
 * - a b c < x a c
 * - a b c < x y z
 */
class PrefixOrder : public Comparator<PrefixOrder> {
  public:

    /**
     * Constructs a comparator capable of comparing two n-grams.
     *
     * @param order Number of words in each n-gram
     */
    explicit PrefixOrder(std::size_t order) : Comparator<PrefixOrder>(order) {}

    /**
     * Compares two n-grams lexicographically, one word at a time,
     * beginning with the first word of each n-gram and ending with the last word of each n-gram.
     *
     * @param lhs A pointer to the n-gram on the left-hand side of the comparison
     * @param rhs A pointer to the n-gram on the right-hand side of the comparison
     */
    inline bool Compare(const WordIndex *lhs, const WordIndex *rhs) const {
      for (std::size_t i = 0; i < order_; ++i) {
        if (lhs[i] != rhs[i])
          return lhs[i] < rhs[i];
      }
      return false;
    }

    static const unsigned kMatchOffset = 0;
};

template <class Range> struct SuffixLexicographicLess : public std::binary_function<const Range, const Range, bool> {
  bool operator()(const Range first, const Range second) const {
    for (const WordIndex *f = first.end() - 1, *s = second.end() - 1; f >= first.begin() && s >= second.begin(); --f, --s) {
      if (*f < *s) return true;
      if (*f > *s) return false;
    }
    return first.size() < second.size();
  }
};

} // namespace lm

#endif // LM_COMMON_COMPARE_H
