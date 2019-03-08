// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Tuple weight set operation definitions.

#ifndef FST_TUPLE_WEIGHT_H_
#define FST_TUPLE_WEIGHT_H_

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/weight.h>


namespace fst {

// n-tuple weight, element of the n-th Cartesian power of W.
template <class W, size_t n>
class TupleWeight {
 public:
  using ReverseWeight = TupleWeight<typename W::ReverseWeight, n>;

  using Weight = W;
  using Index = size_t;

  TupleWeight(const TupleWeight &other) { values_ = other.values_; }

  TupleWeight<W, n> &operator=(const TupleWeight<W, n> &other) {
    values_ = other.values_;
    return *this;
  }

  template <class Iterator>
  TupleWeight(Iterator begin, Iterator end) {
    std::copy(begin, end, values_.begin());
  }

  explicit TupleWeight(const W &weight = W::Zero()) { values_.fill(weight); }

  // Initialize component `index` to `weight`; initialize all other components
  // to `default_weight`
  TupleWeight(Index index, const W &weight, const W &default_weight)
      : TupleWeight(default_weight) {
    values_[index] = weight;
  }

  static const TupleWeight<W, n> &Zero() {
    static const TupleWeight<W, n> zero(W::Zero());
    return zero;
  }

  static const TupleWeight<W, n> &One() {
    static const TupleWeight<W, n> one(W::One());
    return one;
  }

  static const TupleWeight<W, n> &NoWeight() {
    static const TupleWeight<W, n> no_weight(W::NoWeight());
    return no_weight;
  }

  constexpr static size_t Length() { return n; }

  std::istream &Read(std::istream &istrm) {
    for (size_t i = 0; i < n; ++i) values_[i].Read(istrm);
    return istrm;
  }

  std::ostream &Write(std::ostream &ostrm) const {
    for (size_t i = 0; i < n; ++i) values_[i].Write(ostrm);
    return ostrm;
  }

  bool Member() const {
    return std::all_of(values_.begin(), values_.end(),
                       std::mem_fn(&W::Member));
  }

  size_t Hash() const {
    uint64_t hash = 0;
    for (size_t i = 0; i < n; ++i) hash = 5 * hash + values_[i].Hash();
    return size_t(hash);
  }

  TupleWeight<W, n> Quantize(float delta = kDelta) const {
    TupleWeight<W, n> weight;
    for (size_t i = 0; i < n; ++i) {
      weight.values_[i] = values_[i].Quantize(delta);
    }
    return weight;
  }

  ReverseWeight Reverse() const {
    TupleWeight<W, n> w;
    for (size_t i = 0; i < n; ++i) w.values_[i] = values_[i].Reverse();
    return w;
  }

  const W &Value(size_t i) const { return values_[i]; }

  void SetValue(size_t i, const W &w) { values_[i] = w; }

 private:
  std::array<W, n> values_;
};

template <class W, size_t n>
inline bool operator==(const TupleWeight<W, n> &w1,
                       const TupleWeight<W, n> &w2) {
  for (size_t i = 0; i < n; ++i) {
    if (w1.Value(i) != w2.Value(i)) return false;
  }
  return true;
}

template <class W, size_t n>
inline bool operator!=(const TupleWeight<W, n> &w1,
                       const TupleWeight<W, n> &w2) {
  for (size_t i = 0; i < n; ++i) {
    if (w1.Value(i) != w2.Value(i)) return true;
  }
  return false;
}

template <class W, size_t n>
inline bool ApproxEqual(const TupleWeight<W, n> &w1,
                        const TupleWeight<W, n> &w2, float delta = kDelta) {
  for (size_t i = 0; i < n; ++i) {
    if (!ApproxEqual(w1.Value(i), w2.Value(i), delta)) return false;
  }
  return true;
}

template <class W, size_t n>
inline std::ostream &operator<<(std::ostream &strm,
                                const TupleWeight<W, n> &w) {
  CompositeWeightWriter writer(strm);
  writer.WriteBegin();
  for (size_t i = 0; i < n; ++i) writer.WriteElement(w.Value(i));
  writer.WriteEnd();
  return strm;
}

template <class W, size_t n>
inline std::istream &operator>>(std::istream &strm, TupleWeight<W, n> &w) {
  CompositeWeightReader reader(strm);
  reader.ReadBegin();
  W v;
  // Reads first n-1 elements.
  static_assert(n > 0, "Size must be positive.");
  for (size_t i = 0; i < n - 1; ++i) {
    reader.ReadElement(&v);
    w.SetValue(i, v);
  }
  // Reads n-th element.
  reader.ReadElement(&v, true);
  w.SetValue(n - 1, v);
  reader.ReadEnd();
  return strm;
}

}  // namespace fst

#endif  // FST_TUPLE_WEIGHT_H_
