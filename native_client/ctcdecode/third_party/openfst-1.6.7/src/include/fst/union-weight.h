// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Union weight set and associated semiring operation definitions.
//
// TODO(riley): add in normalizer functor

#ifndef FST_UNION_WEIGHT_H_
#define FST_UNION_WEIGHT_H_

#include <cstdlib>

#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <utility>

#include <fst/weight.h>


namespace fst {

// Example UnionWeightOptions for UnionWeight template below. The Merge
// operation is used to collapse elements of the set and the Compare function
// to efficiently implement the merge. In the simplest case, merge would just
// apply with equality of set elements so the result is a set (and not a
// multiset). More generally, this can be used to maintain the multiplicity or
// other such weight associated with the set elements (cf. Gallic weights).

// template <class W>
// struct UnionWeightOptions {
//   // Comparison function C is a total order on W that is monotonic w.r.t. to
//   // Times: for all a, b,c != Zero(): C(a, b) => C(ca, cb) and is
//   // anti-monotonic w.r.rt to Divide: C(a, b) => C(c/b, c/a).
//   //
//   // For all a, b: only one of C(a, b), C(b, a) or a ~ b must true where
//   // ~ is an equivalence relation on W. Also we require a ~ b iff
//   // a.Reverse() ~ b.Reverse().
//   using Compare = NaturalLess<W>;
//
//   // How to combine two weights if a ~ b as above. For all a, b: a ~ b =>
//   // merge(a, b) ~ a, Merge must define a semiring endomorphism from the
//   // unmerged weight sets to the merged weight sets.
//   struct Merge {
//     W operator()(const W &w1, const W &w2) const { return w1; }
//   };
//
//   // For ReverseWeight.
//   using ReverseOptions = UnionWeightOptions<ReverseWeight>;
// };

template <class W, class O>
class UnionWeight;

template <class W, class O>
class UnionWeightIterator;

template <class W, class O>
class UnionWeightReverseIterator;

template <class W, class O>
bool operator==(const UnionWeight<W, O> &, const UnionWeight<W, O> &);

// Semiring that uses Times() and One() from W and union and the empty set
// for Plus() and Zero(), respectively. Template argument O specifies the union
// weight options as above.
template <class W, class O>
class UnionWeight {
 public:
  using Weight = W;
  using Compare = typename O::Compare;
  using Merge = typename O::Merge;

  using ReverseWeight =
      UnionWeight<typename W::ReverseWeight, typename O::ReverseOptions>;

  friend class UnionWeightIterator<W, O>;
  friend class UnionWeightReverseIterator<W, O>;
  friend bool operator==
      <>(const UnionWeight<W, O> &, const UnionWeight<W, O> &);

  // Sets represented as first_ weight + rest_ weights. Uses first_ as
  // NoWeight() to indicate the union weight Zero() ask the empty set. Uses
  // rest_ containing NoWeight() to indicate the union weight NoWeight().
  UnionWeight() : first_(W::NoWeight()) {}

  explicit UnionWeight(W weight) : first_(weight) {
    if (weight == W::NoWeight()) rest_.push_back(weight);
  }

  static const UnionWeight<W, O> &Zero() {
    static const UnionWeight<W, O> zero(W::NoWeight());
    return zero;
  }

  static const UnionWeight<W, O> &One() {
    static const UnionWeight<W, O> one(W::One());
    return one;
  }

  static const UnionWeight<W, O> &NoWeight() {
    static const UnionWeight<W, O> no_weight(W::Zero(), W::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type = new string(W::Type() + "_union");
    return *type;
  }

  static constexpr uint64 Properties() {
    return W::Properties() &
           (kLeftSemiring | kRightSemiring | kCommutative | kIdempotent);
  }

  bool Member() const;

  std::istream &Read(std::istream &strm);

  std::ostream &Write(std::ostream &strm) const;

  size_t Hash() const;

  UnionWeight<W, O> Quantize(float delta = kDelta) const;

  ReverseWeight Reverse() const;

  // These operations combined with the UnionWeightIterator and
  // UnionWeightReverseIterator provide the access and mutation of the union
  // weight internal elements.

  // Common initializer among constructors; clears existing UnionWeight.
  void Clear() {
    first_ = W::NoWeight();
    rest_.clear();
  }

  size_t Size() const { return first_.Member() ? rest_.size() + 1 : 0; }

  const W &Back() const { return rest_.empty() ? first_ : rest_.back(); }

  // When srt is true, assumes elements added sorted w.r.t Compare and merging
  // of weights performed as needed. Otherwise, just ensures first_ is the
  // least element wrt Compare.
  void PushBack(W weight, bool srt);

  // Sorts the elements of the set. Assumes that first_, if present, is the
  // least element.
  void Sort() { rest_.sort(comp_); }

 private:
  W &Back() {
    if (rest_.empty()) {
      return first_;
    } else {
      return rest_.back();
    }
  }

  UnionWeight(W w1, W w2) : first_(std::move(w1)), rest_(1, std::move(w2)) {}

  W first_;            // First weight in set.
  std::list<W> rest_;  // Remaining weights in set.
  Compare comp_;
  Merge merge_;
};

template <class W, class O>
void UnionWeight<W, O>::PushBack(W weight, bool srt) {
  if (!weight.Member()) {
    rest_.push_back(std::move(weight));
  } else if (!first_.Member()) {
    first_ = std::move(weight);
  } else if (srt) {
    auto &back = Back();
    if (comp_(back, weight)) {
      rest_.push_back(std::move(weight));
    } else {
      back = merge_(back, std::move(weight));
    }
  } else {
    if (comp_(first_, weight)) {
      rest_.push_back(std::move(weight));
    } else {
      rest_.push_back(first_);
      first_ = std::move(weight);
    }
  }
}

// Traverses union weight in the forward direction.
template <class W, class O>
class UnionWeightIterator {
 public:
  explicit UnionWeightIterator(const UnionWeight<W, O> &weight)
      : first_(weight.first_),
        rest_(weight.rest_),
        init_(true),
        it_(rest_.begin()) {}

  bool Done() const { return init_ ? !first_.Member() : it_ == rest_.end(); }

  const W &Value() const { return init_ ? first_ : *it_; }

  void Next() {
    if (init_) {
      init_ = false;
    } else {
      ++it_;
    }
  }

  void Reset() {
    init_ = true;
    it_ = rest_.begin();
  }

 private:
  const W &first_;
  const std::list<W> &rest_;
  bool init_;  // in the initialized state?
  typename std::list<W>::const_iterator it_;
};

// Traverses union weight in backward direction.
template <typename L, class O>
class UnionWeightReverseIterator {
 public:
  explicit UnionWeightReverseIterator(const UnionWeight<L, O> &weight)
      : first_(weight.first_),
        rest_(weight.rest_),
        fin_(!first_.Member()),
        it_(rest_.rbegin()) {}

  bool Done() const { return fin_; }

  const L &Value() const { return it_ == rest_.rend() ? first_ : *it_; }

  void Next() {
    if (it_ == rest_.rend()) {
      fin_ = true;
    } else {
      ++it_;
    }
  }

  void Reset() {
    fin_ = !first_.Member();
    it_ = rest_.rbegin();
  }

 private:
  const L &first_;
  const std::list<L> &rest_;
  bool fin_;  // in the final state?
  typename std::list<L>::const_reverse_iterator it_;
};

// UnionWeight member functions follow that require UnionWeightIterator.
template <class W, class O>
inline std::istream &UnionWeight<W, O>::Read(std::istream &istrm) {
  Clear();
  int32 size;
  ReadType(istrm, &size);
  for (int i = 0; i < size; ++i) {
    W weight;
    ReadType(istrm, &weight);
    PushBack(weight, true);
  }
  return istrm;
}

template <class W, class O>
inline std::ostream &UnionWeight<W, O>::Write(std::ostream &ostrm) const {
  const int32 size = Size();
  WriteType(ostrm, size);
  for (UnionWeightIterator<W, O> it(*this); !it.Done(); it.Next()) {
    WriteType(ostrm, it.Value());
  }
  return ostrm;
}

template <class W, class O>
inline bool UnionWeight<W, O>::Member() const {
  if (Size() <= 1) return true;
  for (UnionWeightIterator<W, O> it(*this); !it.Done(); it.Next()) {
    if (!it.Value().Member()) return false;
  }
  return true;
}

template <class W, class O>
inline UnionWeight<W, O> UnionWeight<W, O>::Quantize(float delta) const {
  UnionWeight<W, O> weight;
  for (UnionWeightIterator<W, O> it(*this); !it.Done(); it.Next()) {
    weight.PushBack(it.Value().Quantize(delta), true);
  }
  return weight;
}

template <class W, class O>
inline typename UnionWeight<W, O>::ReverseWeight UnionWeight<W, O>::Reverse()
    const {
  ReverseWeight weight;
  for (UnionWeightIterator<W, O> it(*this); !it.Done(); it.Next()) {
    weight.PushBack(it.Value().Reverse(), false);
  }
  weight.Sort();
  return weight;
}

template <class W, class O>
inline size_t UnionWeight<W, O>::Hash() const {
  size_t h = 0;
  static constexpr int lshift = 5;
  static constexpr int rshift = CHAR_BIT * sizeof(size_t) - lshift;
  for (UnionWeightIterator<W, O> it(*this); !it.Done(); it.Next()) {
    h = h << lshift ^ h >> rshift ^ it.Value().Hash();
  }
  return h;
}

// Requires union weight has been canonicalized.
template <class W, class O>
inline bool operator==(const UnionWeight<W, O> &w1,
                       const UnionWeight<W, O> &w2) {
  if (w1.Size() != w2.Size()) return false;
  UnionWeightIterator<W, O> it1(w1);
  UnionWeightIterator<W, O> it2(w2);
  for (; !it1.Done(); it1.Next(), it2.Next()) {
    if (it1.Value() != it2.Value()) return false;
  }
  return true;
}

// Requires union weight has been canonicalized.
template <class W, class O>
inline bool operator!=(const UnionWeight<W, O> &w1,
                       const UnionWeight<W, O> &w2) {
  return !(w1 == w2);
}

// Requires union weight has been canonicalized.
template <class W, class O>
inline bool ApproxEqual(const UnionWeight<W, O> &w1,
                        const UnionWeight<W, O> &w2, float delta = kDelta) {
  if (w1.Size() != w2.Size()) return false;
  UnionWeightIterator<W, O> it1(w1);
  UnionWeightIterator<W, O> it2(w2);
  for (; !it1.Done(); it1.Next(), it2.Next()) {
    if (!ApproxEqual(it1.Value(), it2.Value(), delta)) return false;
  }
  return true;
}

template <class W, class O>
inline std::ostream &operator<<(std::ostream &ostrm,
                                const UnionWeight<W, O> &weight) {
  UnionWeightIterator<W, O> it(weight);
  if (it.Done()) {
    return ostrm << "EmptySet";
  } else if (!weight.Member()) {
    return ostrm << "BadSet";
  } else {
    CompositeWeightWriter writer(ostrm);
    writer.WriteBegin();
    for (; !it.Done(); it.Next()) writer.WriteElement(it.Value());
    writer.WriteEnd();
  }
  return ostrm;
}

template <class W, class O>
inline std::istream &operator>>(std::istream &istrm,
                                UnionWeight<W, O> &weight) {
  string s;
  istrm >> s;
  if (s == "EmptySet") {
    weight = UnionWeight<W, O>::Zero();
  } else if (s == "BadSet") {
    weight = UnionWeight<W, O>::NoWeight();
  } else {
    weight = UnionWeight<W, O>::Zero();
    std::istringstream sstrm(s);
    CompositeWeightReader reader(sstrm);
    reader.ReadBegin();
    bool more = true;
    while (more) {
      W v;
      more = reader.ReadElement(&v);
      weight.PushBack(v, true);
    }
    reader.ReadEnd();
  }
  return istrm;
}

template <class W, class O>
inline UnionWeight<W, O> Plus(const UnionWeight<W, O> &w1,
                              const UnionWeight<W, O> &w2) {
  if (!w1.Member() || !w2.Member()) return UnionWeight<W, O>::NoWeight();
  if (w1 == UnionWeight<W, O>::Zero()) return w2;
  if (w2 == UnionWeight<W, O>::Zero()) return w1;
  UnionWeightIterator<W, O> it1(w1);
  UnionWeightIterator<W, O> it2(w2);
  UnionWeight<W, O> sum;
  typename O::Compare comp;
  while (!it1.Done() && !it2.Done()) {
    const auto v1 = it1.Value();
    const auto v2 = it2.Value();
    if (comp(v1, v2)) {
      sum.PushBack(v1, true);
      it1.Next();
    } else {
      sum.PushBack(v2, true);
      it2.Next();
    }
  }
  for (; !it1.Done(); it1.Next()) sum.PushBack(it1.Value(), true);
  for (; !it2.Done(); it2.Next()) sum.PushBack(it2.Value(), true);
  return sum;
}

template <class W, class O>
inline UnionWeight<W, O> Times(const UnionWeight<W, O> &w1,
                               const UnionWeight<W, O> &w2) {
  if (!w1.Member() || !w2.Member()) return UnionWeight<W, O>::NoWeight();
  if (w1 == UnionWeight<W, O>::Zero() || w2 == UnionWeight<W, O>::Zero()) {
    return UnionWeight<W, O>::Zero();
  }
  UnionWeightIterator<W, O> it1(w1);
  UnionWeightIterator<W, O> it2(w2);
  UnionWeight<W, O> prod1;
  for (; !it1.Done(); it1.Next()) {
    UnionWeight<W, O> prod2;
    for (; !it2.Done(); it2.Next()) {
      prod2.PushBack(Times(it1.Value(), it2.Value()), true);
    }
    prod1 = Plus(prod1, prod2);
    it2.Reset();
  }
  return prod1;
}

template <class W, class O>
inline UnionWeight<W, O> Divide(const UnionWeight<W, O> &w1,
                                const UnionWeight<W, O> &w2, DivideType typ) {
  if (!w1.Member() || !w2.Member()) return UnionWeight<W, O>::NoWeight();
  if (w1 == UnionWeight<W, O>::Zero() || w2 == UnionWeight<W, O>::Zero()) {
    return UnionWeight<W, O>::Zero();
  }
  UnionWeightIterator<W, O> it1(w1);
  UnionWeightReverseIterator<W, O> it2(w2);
  UnionWeight<W, O> quot;
  if (w1.Size() == 1) {
    for (; !it2.Done(); it2.Next()) {
      quot.PushBack(Divide(it1.Value(), it2.Value(), typ), true);
    }
  } else if (w2.Size() == 1) {
    for (; !it1.Done(); it1.Next()) {
      quot.PushBack(Divide(it1.Value(), it2.Value(), typ), true);
    }
  } else {
    quot = UnionWeight<W, O>::NoWeight();
  }
  return quot;
}

// This function object generates weights over the union of weights for the
// underlying generators for the template weight types. This is intended
// primarily for testing.
template <class W, class O>
class WeightGenerate<UnionWeight<W, O>> {
 public:
  using Weight = UnionWeight<W, O>;
  using Generate = WeightGenerate<W>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : generate_(false), allow_zero_(allow_zero),
        num_random_weights_(num_random_weights) {}

  Weight operator()() const {
    const int n = rand() % (num_random_weights_ + 1);  // NOLINT
    if (allow_zero_ && n == num_random_weights_) {
      return Weight::Zero();
    } else if (n % 2 == 0) {
      return Weight(generate_());
    } else {
      return Plus(Weight(generate_()), Weight(generate_()));
    }
  }

 private:
  Generate generate_;
  // Permits Zero() and zero divisors.
  bool allow_zero_;
  // The number of alternative random weights.
  const size_t num_random_weights_;
};

}  // namespace fst

#endif  // FST_UNION_WEIGHT_H_
