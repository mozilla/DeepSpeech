// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Product weight set and associated semiring operation definitions.

#ifndef FST_PRODUCT_WEIGHT_H_
#define FST_PRODUCT_WEIGHT_H_

#include <string>
#include <utility>

#include <fst/pair-weight.h>
#include <fst/weight.h>


namespace fst {

// Product semiring: W1 * W2.
template <class W1, class W2>
class ProductWeight : public PairWeight<W1, W2> {
 public:
  using ReverseWeight =
      ProductWeight<typename W1::ReverseWeight, typename W2::ReverseWeight>;

  ProductWeight() {}

  explicit ProductWeight(const PairWeight<W1, W2> &weight)
      : PairWeight<W1, W2>(weight) {}

  ProductWeight(W1 w1, W2 w2)
      : PairWeight<W1, W2>(std::move(w1), std::move(w2)) {}

  static const ProductWeight &Zero() {
    static const ProductWeight zero(PairWeight<W1, W2>::Zero());
    return zero;
  }

  static const ProductWeight &One() {
    static const ProductWeight one(PairWeight<W1, W2>::One());
    return one;
  }

  static const ProductWeight &NoWeight() {
    static const ProductWeight no_weight(PairWeight<W1, W2>::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type =
        new string(W1::Type() + "_X_" + W2::Type());
    return *type;
  }

  static constexpr uint64 Properties() {
    return W1::Properties() & W2::Properties() &
           (kLeftSemiring | kRightSemiring | kCommutative | kIdempotent);
  }

  ProductWeight Quantize(float delta = kDelta) const {
    return ProductWeight(PairWeight<W1, W2>::Quantize(delta));
  }

  ReverseWeight Reverse() const {
    return ReverseWeight(PairWeight<W1, W2>::Reverse());
  }
};

template <class W1, class W2>
inline ProductWeight<W1, W2> Plus(const ProductWeight<W1, W2> &w1,
                                  const ProductWeight<W1, W2> &w2) {
  return ProductWeight<W1, W2>(Plus(w1.Value1(), w2.Value1()),
                               Plus(w1.Value2(), w2.Value2()));
}

template <class W1, class W2>
inline ProductWeight<W1, W2> Times(const ProductWeight<W1, W2> &w1,
                                   const ProductWeight<W1, W2> &w2) {
  return ProductWeight<W1, W2>(Times(w1.Value1(), w2.Value1()),
                               Times(w1.Value2(), w2.Value2()));
}

template <class W1, class W2>
inline ProductWeight<W1, W2> Divide(const ProductWeight<W1, W2> &w1,
                                    const ProductWeight<W1, W2> &w2,
                                    DivideType typ = DIVIDE_ANY) {
  return ProductWeight<W1, W2>(Divide(w1.Value1(), w2.Value1(), typ),
                               Divide(w1.Value2(), w2.Value2(), typ));
}

// This function object generates weights by calling the underlying generators
// for the template weight types, like all other pair weight types. This is
// intended primarily for testing.
template <class W1, class W2>
class WeightGenerate<ProductWeight<W1, W2>> :
    public WeightGenerate<PairWeight<W1, W2>> {
 public:
  using Weight = ProductWeight<W1, W2>;
  using Generate = WeightGenerate<PairWeight<W1, W2>>;

  explicit WeightGenerate(bool allow_zero = true) : Generate(allow_zero) {}

  Weight operator()() const { return Weight(Generate::operator()()); }
};

}  // namespace fst

#endif  // FST_PRODUCT_WEIGHT_H_
