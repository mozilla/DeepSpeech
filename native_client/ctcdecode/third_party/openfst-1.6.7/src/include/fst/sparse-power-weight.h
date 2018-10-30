// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Cartesian power weight semiring operation definitions, using
// SparseTupleWeight as underlying representation.

#ifndef FST_SPARSE_POWER_WEIGHT_H_
#define FST_SPARSE_POWER_WEIGHT_H_

#include <climits>
#include <string>

#include <fst/sparse-tuple-weight.h>
#include <fst/weight.h>


namespace fst {

// Sparse cartesian power semiring: W ^ n
//
// Forms:
//
//  - a left semimodule when W is a left semiring,
//  - a right semimodule when W is a right semiring,
//  - a bisemimodule when W is a semiring,
//    the free semimodule of rank n over W
//
// The Times operation is overloaded to provide the left and right scalar
// products.
//
// K is the key value type. kNoKey (-1) is reserved for internal use
template <class W, class K = int>
class SparsePowerWeight : public SparseTupleWeight<W, K> {
 public:
  using ReverseWeight = SparsePowerWeight<typename W::ReverseWeight, K>;

  SparsePowerWeight() {}

  explicit SparsePowerWeight(const SparseTupleWeight<W, K> &weight)
      : SparseTupleWeight<W, K>(weight) {}

  template <class Iterator>
  SparsePowerWeight(Iterator begin, Iterator end)
      : SparseTupleWeight<W, K>(begin, end) {}

  // Initialize component `key` to `weight`, with `default_weight` for all
  // other components.
  SparsePowerWeight(const K &key, const W &weight,
                    const W &default_weight = W::Zero())
      : SparseTupleWeight<W, K>(key, weight, default_weight) {}

  static const SparsePowerWeight &Zero() {
    static const SparsePowerWeight zero(SparseTupleWeight<W, K>::Zero());
    return zero;
  }

  static const SparsePowerWeight &One() {
    static const SparsePowerWeight one(SparseTupleWeight<W, K>::One());
    return one;
  }

  static const SparsePowerWeight &NoWeight() {
    static const SparsePowerWeight no_weight(
        SparseTupleWeight<W, K>::NoWeight());
    return no_weight;
  }

  // Overide this: Overwrite the Type method to reflect the key type if using
  // a non-default key type.
  static const string &Type() {
    static const string *const type = [] {
      string type = W::Type() + "_^n";
      if (sizeof(K) != sizeof(uint32)) {
        type += "_" + std::to_string(CHAR_BIT * sizeof(K));
      }
      return new string(type);
    }();
    return *type;
  }

  static constexpr uint64 Properties() {
    return W::Properties() &
           (kLeftSemiring | kRightSemiring | kCommutative | kIdempotent);
  }

  SparsePowerWeight Quantize(float delta = kDelta) const {
    return SparsePowerWeight(SparseTupleWeight<W, K>::Quantize(delta));
  }

  ReverseWeight Reverse() const {
    return ReverseWeight(SparseTupleWeight<W, K>::Reverse());
  }
};

template <class W, class K, class M>
inline SparsePowerWeight<W, K> SparsePowerWeightMap(
    const SparsePowerWeight<W, K> &w1,
    const SparsePowerWeight<W, K> &w2,
    const M &operator_mapper) {
  SparsePowerWeight<W, K> result;
  SparseTupleWeightMap(&result, w1, w2, operator_mapper);
  return result;
}

// Semimodule plus operation.
template <class W, class K>
inline SparsePowerWeight<W, K> Plus(const SparsePowerWeight<W, K> &w1,
                                    const SparsePowerWeight<W, K> &w2) {
  return SparsePowerWeightMap(w1, w2, [](const K &k, const W &v1, const W &v2) {
    return Plus(v1, v2);
  });
}

// Semimodule times operation.
template <class W, class K>
inline SparsePowerWeight<W, K> Times(const SparsePowerWeight<W, K> &w1,
                                     const SparsePowerWeight<W, K> &w2) {
  return SparsePowerWeightMap(w1, w2, [](const K &k, const W &v1, const W &v2) {
    return Times(v1, v2);
  });
}

// Semimodule divide operation.
template <class W, class K>
inline SparsePowerWeight<W, K> Divide(const SparsePowerWeight<W, K> &w1,
                                      const SparsePowerWeight<W, K> &w2,
                                      DivideType type = DIVIDE_ANY) {
  return SparsePowerWeightMap(w1, w2,
                              [type](const K &k, const W &v1, const W &v2) {
                                return Divide(v1, v2, type);
                              });
}

// Semimodule dot product operation.
template <class W, class K>
inline const W &DotProduct(const SparsePowerWeight<W, K> &w1,
                           const SparsePowerWeight<W, K> &w2) {
  const SparsePowerWeight<W, K> product = Times(w1, w2);
  W result(W::Zero());
  for (SparseTupleWeightIterator<W, K> it(product); !it.Done(); it.Next()) {
    result = Plus(result, it.Value().second);
  }
  return result;
}

template <class W, class K>
inline bool ApproxEqual(const SparsePowerWeight<W, K> &w1,
                        const SparsePowerWeight<W, K> &w2,
                        float delta = kDelta) {
  auto result = SparsePowerWeightMap(
      w1, w2, [delta](const K &k, const W &v1, const W &v2) {
        return ApproxEqual(v1, v2, delta) ? W::One() : W::Zero();
      });
  return result == SparsePowerWeight<W, K>::One();
}

template <class W, class K>
inline SparsePowerWeight<W, K> Times(const W &k,
                                     const SparsePowerWeight<W, K> &w2) {
  const SparseTupleWeight<W, K> t2(k);
  const SparsePowerWeight<W, K> w1(t2);
  return Times(w1, w2);
}

template <class W, class K>
inline SparsePowerWeight<W, K> Times(const SparsePowerWeight<W, K> &w1,
                                     const W &k) {
  const SparseTupleWeight<W, K> t2(k);
  const SparsePowerWeight<W, K> w2(t2);
  return Times(w1, w2);
}

template <class W, class K>
inline SparsePowerWeight<W, K> Divide(const SparsePowerWeight<W, K> &w1,
                                      const W &k,
                                      DivideType divide_type = DIVIDE_ANY) {
  const SparseTupleWeight<W, K> t2(k);
  const SparsePowerWeight<W, K> w2(t2);
  return Divide(w1, w2, divide_type);
}

// This function object generates weights over the Cartesian power of rank
// n over the underlying weight. This is intended primarily for testing.
template <class W, class K>
class WeightGenerate<SparsePowerWeight<W, K>> {
 public:
  using Weight = SparsePowerWeight<W, K>;
  using Generate = WeightGenerate<W>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t sparse_power_rank = 3)
      : generate_(allow_zero), sparse_power_rank_(sparse_power_rank) {}

  Weight operator()() const {
    Weight weight;
    for (size_t i = 1; i <= sparse_power_rank_; ++i) {
      weight.PushBack(i, generate_(), true);
    }
    return weight;
  }

 private:
  const Generate generate_;
  const size_t sparse_power_rank_;
};

}  // namespace fst

#endif  // FST_SPARSE_POWER_WEIGHT_H_
