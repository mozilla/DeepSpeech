// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Cartesian power weight semiring operation definitions.

#ifndef FST_POWER_WEIGHT_H_
#define FST_POWER_WEIGHT_H_

#include <string>

#include <fst/tuple-weight.h>
#include <fst/weight.h>


namespace fst {

// Cartesian power semiring: W ^ n
//
// Forms:
//  - a left semimodule when W is a left semiring,
//  - a right semimodule when W is a right semiring,
//  - a bisemimodule when W is a semiring,
//    the free semimodule of rank n over W
// The Times operation is overloaded to provide the left and right scalar
// products.
template <class W, size_t n>
class PowerWeight : public TupleWeight<W, n> {
 public:
  using ReverseWeight = PowerWeight<typename W::ReverseWeight, n>;

  PowerWeight() {}

  explicit PowerWeight(const TupleWeight<W, n> &weight)
      : TupleWeight<W, n>(weight) {}

  template <class Iterator>
  PowerWeight(Iterator begin, Iterator end) : TupleWeight<W, n>(begin, end) {}

  // Initialize component `index` to `weight`; initialize all other components
  // to `default_weight`
  PowerWeight(size_t index, const W &weight,
              const W &default_weight = W::Zero())
      : TupleWeight<W, n>(index, weight, default_weight) {}

  static const PowerWeight &Zero() {
    static const PowerWeight zero(TupleWeight<W, n>::Zero());
    return zero;
  }

  static const PowerWeight &One() {
    static const PowerWeight one(TupleWeight<W, n>::One());
    return one;
  }

  static const PowerWeight &NoWeight() {
    static const PowerWeight no_weight(TupleWeight<W, n>::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type =
        new string(W::Type() + "_^" + std::to_string(n));
    return *type;
  }

  static constexpr uint64_t Properties() {
    return W::Properties() &
           (kLeftSemiring | kRightSemiring | kCommutative | kIdempotent);
  }

  PowerWeight Quantize(float delta = kDelta) const {
    return PowerWeight(TupleWeight<W, n>::Quantize(delta));
  }

  ReverseWeight Reverse() const {
    return ReverseWeight(TupleWeight<W, n>::Reverse());
  }
};

// Semiring plus operation.
template <class W, size_t n>
inline PowerWeight<W, n> Plus(const PowerWeight<W, n> &w1,
                              const PowerWeight<W, n> &w2) {
  PowerWeight<W, n> result;
  for (size_t i = 0; i < n; ++i) {
    result.SetValue(i, Plus(w1.Value(i), w2.Value(i)));
  }
  return result;
}

// Semiring times operation.
template <class W, size_t n>
inline PowerWeight<W, n> Times(const PowerWeight<W, n> &w1,
                               const PowerWeight<W, n> &w2) {
  PowerWeight<W, n> result;
  for (size_t i = 0; i < n; ++i) {
    result.SetValue(i, Times(w1.Value(i), w2.Value(i)));
  }
  return result;
}

// Semiring divide operation.
template <class W, size_t n>
inline PowerWeight<W, n> Divide(const PowerWeight<W, n> &w1,
                                const PowerWeight<W, n> &w2,
                                DivideType type = DIVIDE_ANY) {
  PowerWeight<W, n> result;
  for (size_t i = 0; i < n; ++i) {
    result.SetValue(i, Divide(w1.Value(i), w2.Value(i), type));
  }
  return result;
}

// Semimodule left scalar product.
template <class W, size_t n>
inline PowerWeight<W, n> Times(const W &scalar,
                               const PowerWeight<W, n> &weight) {
  PowerWeight<W, n> result;
  for (size_t i = 0; i < n; ++i) {
    result.SetValue(i, Times(scalar, weight.Value(i)));
  }
  return result;
}

// Semimodule right scalar product.
template <class W, size_t n>
inline PowerWeight<W, n> Times(const PowerWeight<W, n> &weight,
                               const W &scalar) {
  PowerWeight<W, n> result;
  for (size_t i = 0; i < n; ++i) {
    result.SetValue(i, Times(weight.Value(i), scalar));
  }
  return result;
}

// Semimodule dot product.
template <class W, size_t n>
inline W DotProduct(const PowerWeight<W, n> &w1, const PowerWeight<W, n> &w2) {
  W result(W::Zero());
  for (size_t i = 0; i < n; ++i) {
    result = Plus(result, Times(w1.Value(i), w2.Value(i)));
  }
  return result;
}

// This function object generates weights over the Cartesian power of rank
// n over the underlying weight. This is intended primarily for testing.
template <class W, size_t n>
class WeightGenerate<PowerWeight<W, n>> {
 public:
  using Weight = PowerWeight<W, n>;
  using Generate = WeightGenerate<W>;

  explicit WeightGenerate(bool allow_zero = true) : generate_(allow_zero) {}

  Weight operator()() const {
    Weight result;
    for (size_t i = 0; i < n; ++i) result.SetValue(i, generate_());
    return result;
  }

 private:
  Generate generate_;
};

}  // namespace fst

#endif  // FST_POWER_WEIGHT_H_
