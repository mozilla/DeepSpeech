// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Expectation semiring as described by Jason Eisner:
// See: doi=10.1.1.22.9398
// Multiplex semiring operations and identities:
//    One: <One, Zero>
//    Zero: <Zero, Zero>
//    Plus: <a1, b1> + <a2, b2> = < (a1 + a2) , (b1 + b2) >
//    Times: <a1, b1> * <a2, b2> = < (a1 * a2) , [(a1 * b2) + (a2 * b1)] >
//    Division: Undefined (currently)
//
// Usually used to store the pair <probability, random_variable> so that
// ShortestDistance[Fst<ArcTpl<ExpectationWeight<P, V>>>]
//    == < PosteriorProbability, Expected_Value[V] >

#ifndef FST_EXPECTATION_WEIGHT_H_
#define FST_EXPECTATION_WEIGHT_H_

#include <string>

#include <fst/log.h>

#include <fst/pair-weight.h>
#include <fst/product-weight.h>


namespace fst {

// X1 is usually a probability weight like LogWeight.
// X2 is usually a random variable or vector (see SignedLogWeight or
// SparsePowerWeight).
//
// If X1 is distinct from X2, it is required that there is an external product
// between X1 and X2 and if both semriring are commutative, or left or right
// semirings, then result must have those properties.
template <class X1, class X2>
class ExpectationWeight : public PairWeight<X1, X2> {
 public:
  using PairWeight<X1, X2>::Value1;
  using PairWeight<X1, X2>::Value2;

  using PairWeight<X1, X2>::Reverse;
  using PairWeight<X1, X2>::Quantize;
  using PairWeight<X1, X2>::Member;

  using ReverseWeight =
      ExpectationWeight<typename X1::ReverseWeight, typename X2::ReverseWeight>;

  ExpectationWeight() : PairWeight<X1, X2>(Zero()) {}

  ExpectationWeight(const ExpectationWeight &weight)
      : PairWeight<X1, X2>(weight) {}

  explicit ExpectationWeight(const PairWeight<X1, X2> &weight)
      : PairWeight<X1, X2>(weight) {}

  ExpectationWeight(const X1 &x1, const X2 &x2) : PairWeight<X1, X2>(x1, x2) {}

  static const ExpectationWeight &Zero() {
    static const ExpectationWeight zero(X1::Zero(), X2::Zero());
    return zero;
  }

  static const ExpectationWeight &One() {
    static const ExpectationWeight one(X1::One(), X2::Zero());
    return one;
  }

  static const ExpectationWeight &NoWeight() {
    static const ExpectationWeight no_weight(X1::NoWeight(), X2::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type =
        new string("expectation_" + X1::Type() + "_" + X2::Type());
    return *type;
  }

  PairWeight<X1, X2> Quantize(float delta = kDelta) const {
    return ExpectationWeight(PairWeight<X1, X2>::Quantize());
  }

  ReverseWeight Reverse() const {
    return ReverseWeight(PairWeight<X1, X2>::Reverse());
  }

  bool Member() const { return PairWeight<X1, X2>::Member(); }

  static constexpr uint64_t Properties() {
    return X1::Properties() & X2::Properties() &
           (kLeftSemiring | kRightSemiring | kCommutative | kIdempotent);
  }
};

template <class X1, class X2>
inline ExpectationWeight<X1, X2> Plus(const ExpectationWeight<X1, X2> &w1,
                                      const ExpectationWeight<X1, X2> &w2) {
  return ExpectationWeight<X1, X2>(Plus(w1.Value1(), w2.Value1()),
                                   Plus(w1.Value2(), w2.Value2()));
}

template <class X1, class X2>
inline ExpectationWeight<X1, X2> Times(const ExpectationWeight<X1, X2> &w1,
                                       const ExpectationWeight<X1, X2> &w2) {
  return ExpectationWeight<X1, X2>(
      Times(w1.Value1(), w2.Value1()),
      Plus(Times(w1.Value1(), w2.Value2()), Times(w1.Value2(), w2.Value1())));
}

template <class X1, class X2>
inline ExpectationWeight<X1, X2> Divide(const ExpectationWeight<X1, X2> &w1,
                                        const ExpectationWeight<X1, X2> &w2,
                                        DivideType typ = DIVIDE_ANY) {
  FSTERROR() << "ExpectationWeight::Divide: Not implemented";
  return ExpectationWeight<X1, X2>::NoWeight();
}

// This function object generates weights by calling the underlying generators
// for the template weight types, like all other pair weight types. This is
// intended primarily for testing.
template <class X1, class X2>
class WeightGenerate<ExpectationWeight<X1, X2>>
    : public WeightGenerate<PairWeight<X1, X2>> {
 public:
  using Weight = ExpectationWeight<X1, X2>;
  using Generate = WeightGenerate<PairWeight<X1, X2>>;

  explicit WeightGenerate(bool allow_zero = true) : Generate(allow_zero) {}

  Weight operator()() const { return Weight(Generate::operator()()); }
};

}  // namespace fst

#endif  // FST_EXPECTATION_WEIGHT_H_
