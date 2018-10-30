// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Lexicographic weight set and associated semiring operation definitions.
//
// A lexicographic weight is a sequence of weights, each of which must have the
// path property and Times() must be (strongly) cancellative
// (for all a,b,c != Zero(): Times(c, a) = Times(c, b) => a = b,
// Times(a, c) = Times(b, c) => a = b).
// The + operation on two weights a and b is the lexicographically
// prior of a and b.

#ifndef FST_LEXICOGRAPHIC_WEIGHT_H_
#define FST_LEXICOGRAPHIC_WEIGHT_H_

#include <cstdlib>

#include <string>

#include <fst/log.h>

#include <fst/pair-weight.h>
#include <fst/weight.h>


namespace fst {

template <class W1, class W2>
class LexicographicWeight : public PairWeight<W1, W2> {
 public:
  using ReverseWeight = LexicographicWeight<typename W1::ReverseWeight,
                                            typename W2::ReverseWeight>;

  using PairWeight<W1, W2>::Value1;
  using PairWeight<W1, W2>::Value2;
  using PairWeight<W1, W2>::SetValue1;
  using PairWeight<W1, W2>::SetValue2;
  using PairWeight<W1, W2>::Zero;
  using PairWeight<W1, W2>::One;
  using PairWeight<W1, W2>::NoWeight;
  using PairWeight<W1, W2>::Quantize;
  using PairWeight<W1, W2>::Reverse;

  LexicographicWeight() {}

  explicit LexicographicWeight(const PairWeight<W1, W2> &w)
      : PairWeight<W1, W2>(w) {}

  LexicographicWeight(W1 w1, W2 w2) : PairWeight<W1, W2>(w1, w2) {
    if ((W1::Properties() & kPath) != kPath) {
      FSTERROR() << "LexicographicWeight must "
                 << "have the path property: " << W1::Type();
      SetValue1(W1::NoWeight());
    }
    if ((W2::Properties() & kPath) != kPath) {
      FSTERROR() << "LexicographicWeight must "
                 << "have the path property: " << W2::Type();
      SetValue2(W2::NoWeight());
    }
  }

  static const LexicographicWeight &Zero() {
    static const LexicographicWeight zero(PairWeight<W1, W2>::Zero());
    return zero;
  }

  static const LexicographicWeight &One() {
    static const LexicographicWeight one(PairWeight<W1, W2>::One());
    return one;
  }

  static const LexicographicWeight &NoWeight() {
    static const LexicographicWeight no_weight(PairWeight<W1, W2>::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type =
        new string(W1::Type() + "_LT_" + W2::Type());
    return *type;
  }

  bool Member() const {
    if (!Value1().Member() || !Value2().Member()) return false;
    // Lexicographic weights cannot mix zeroes and non-zeroes.
    if (Value1() == W1::Zero() && Value2() == W2::Zero()) return true;
    if (Value1() != W1::Zero() && Value2() != W2::Zero()) return true;
    return false;
  }

  LexicographicWeight Quantize(float delta = kDelta) const {
    return LexicographicWeight(PairWeight<W1, W2>::Quantize());
  }

  ReverseWeight Reverse() const {
    return ReverseWeight(PairWeight<W1, W2>::Reverse());
  }

  static constexpr uint64 Properties() {
    return W1::Properties() & W2::Properties() &
           (kLeftSemiring | kRightSemiring | kPath | kIdempotent |
            kCommutative);
  }
};

template <class W1, class W2>
inline LexicographicWeight<W1, W2> Plus(const LexicographicWeight<W1, W2> &w,
                                        const LexicographicWeight<W1, W2> &v) {
  if (!w.Member() || !v.Member()) {
    return LexicographicWeight<W1, W2>::NoWeight();
  }
  NaturalLess<W1> less1;
  NaturalLess<W2> less2;
  if (less1(w.Value1(), v.Value1())) return w;
  if (less1(v.Value1(), w.Value1())) return v;
  if (less2(w.Value2(), v.Value2())) return w;
  if (less2(v.Value2(), w.Value2())) return v;
  return w;
}

template <class W1, class W2>
inline LexicographicWeight<W1, W2> Times(const LexicographicWeight<W1, W2> &w,
                                         const LexicographicWeight<W1, W2> &v) {
  return LexicographicWeight<W1, W2>(Times(w.Value1(), v.Value1()),
                                     Times(w.Value2(), v.Value2()));
}

template <class W1, class W2>
inline LexicographicWeight<W1, W2> Divide(const LexicographicWeight<W1, W2> &w,
                                          const LexicographicWeight<W1, W2> &v,
                                          DivideType typ = DIVIDE_ANY) {
  return LexicographicWeight<W1, W2>(Divide(w.Value1(), v.Value1(), typ),
                                     Divide(w.Value2(), v.Value2(), typ));
}

// This function object generates weights by calling the underlying generators
// for the templated weight types, like all other pair weight types. However,
// for lexicographic weights, we cannot generate zeroes for the two subweights
// separately: weights are members iff both members are zero or both members
// are non-zero. This is intended primarily for testing.
template <class W1, class W2>
class WeightGenerate<LexicographicWeight<W1, W2>> {
 public:
  using Weight = LexicographicWeight<W1, W1>;
  using Generate1 = WeightGenerate<W1>;
  using Generate2 = WeightGenerate<W2>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : generator1_(false, num_random_weights),
        generator2_(false, num_random_weights), allow_zero_(allow_zero),
        num_random_weights_(num_random_weights) {}

  Weight operator()() const {
    if (allow_zero_) {
      const int n = rand() % (num_random_weights_ + 1);  // NOLINT
      if (n == num_random_weights_) return Weight(W1::Zero(), W2::Zero());
    }
    return Weight(generator1_(), generator2_());
  }

 private:
  const Generate1 generator1_;
  const Generate2 generator2_;
  // Permits Zero() and zero divisors.
  const bool allow_zero_;
  // The number of alternative random weights.
  const size_t num_random_weights_;
};

}  // namespace fst

#endif  // FST_LEXICOGRAPHIC_WEIGHT_H_
