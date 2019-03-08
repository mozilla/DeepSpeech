// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Weights consisting of sets (of integral Labels) and
// associated semiring operation definitions using intersect
// and union.

#ifndef FST_SET_WEIGHT_H_
#define FST_SET_WEIGHT_H_

#include <cstdlib>

#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include <fst/union-weight.h>
#include <fst/weight.h>


namespace fst {

constexpr int kSetEmpty = 0;         // Label for the empty set.
constexpr int kSetUniv = -1;         // Label for the universal set.
constexpr int kSetBad = -2;          // Label for a non-set.
constexpr char kSetSeparator = '_';  // Label separator in sets.

// Determines whether to use (intersect, union) or (union, intersect)
// as (+, *) for the semiring. SET_INTERSECT_UNION_RESTRICTED is a
// restricted version of (intersect, union) that requires summed
// arguments to be equal (or an error is signalled), useful for
// algorithms that require a unique labelled path weight.  SET_BOOLEAN
// treats all non-Zero() elements as equivalent (with Zero() ==
// UnivSet()), useful for algorithms that don't really depend on the
// detailed sets.
enum SetType { SET_INTERSECT_UNION = 0,
               SET_UNION_INTERSECT = 1,
               SET_INTERSECT_UNION_RESTRICT = 2,
               SET_BOOLEAN = 3 };

template <class>
class SetWeightIterator;

// Set semiring of integral labels.
template <typename Label_, SetType S = SET_INTERSECT_UNION>
class SetWeight {
 public:
  using Label = Label_;
  using ReverseWeight = SetWeight<Label, S>;
  using Iterator = SetWeightIterator<SetWeight>;
  friend class SetWeightIterator<SetWeight>;
  // Allow type-converting copy and move constructors private access.
  template <typename L2, SetType S2>
  friend class SetWeight;

  SetWeight() {}

  // Input should be positive, sorted and unique.
  template <typename Iterator>
  SetWeight(const Iterator &begin, const Iterator &end) {
    for (auto iter = begin; iter != end; ++iter) PushBack(*iter);
  }

  // Input should be positive. (Non-positive value has
  // special internal meaning w.r.t. integral constants above.)
  explicit SetWeight(Label label) { PushBack(label); }

  template <SetType S2>
  explicit SetWeight(const SetWeight<Label, S2> &w)
    : first_(w.first_), rest_(w.rest_) {}

  template <SetType S2>
  explicit SetWeight(SetWeight<Label, S2> &&w)
    : first_(w.first_), rest_(std::move(w.rest_)) { w.Clear(); }

  template <SetType S2>
  SetWeight &operator=(const SetWeight<Label, S2> &w) {
    first_ = w.first_;
    rest_ = w.rest_;
    return *this;
  }

  template <SetType S2>
  SetWeight &operator=(SetWeight<Label, S2> &&w) {
    first_ = w.first_;
    rest_ = std::move(w.rest_);
    w.Clear();
    return *this;
  }

  static const SetWeight &Zero() {
    return S == SET_UNION_INTERSECT ? EmptySet() : UnivSet();
  }

  static const SetWeight &One() {
    return S == SET_UNION_INTERSECT ? UnivSet() : EmptySet();
  }

  static const SetWeight &NoWeight() {
    static const auto *const no_weight = new SetWeight(Label(kSetBad));
    return *no_weight;
  }

  static const string &Type() {
    static const string *const type = new string(
        S == SET_UNION_INTERSECT
        ? "union_intersect_set"
        : (S == SET_INTERSECT_UNION
           ? "intersect_union_set"
           : (S == SET_INTERSECT_UNION_RESTRICT
              ? "restricted_set_intersect_union"
              : "boolean_set")));
    return *type;
  }

  bool Member() const;

  std::istream &Read(std::istream &strm);

  std::ostream &Write(std::ostream &strm) const;

  size_t Hash() const;

  SetWeight Quantize(float delta = kDelta) const { return *this; }

  ReverseWeight Reverse() const;

  static constexpr uint64_t Properties() {
    return kIdempotent | kLeftSemiring | kRightSemiring | kCommutative;
  }

  // These operations combined with the SetWeightIterator
  // provide the access and mutation of the set internal elements.

  // The empty set.
  static const SetWeight &EmptySet() {
    static const auto *const empty = new SetWeight(Label(kSetEmpty));
    return *empty;
  }

  // The univeral set.
  static const SetWeight &UnivSet() {
    static const auto *const univ = new SetWeight(Label(kSetUniv));
    return *univ;
  }

  // Clear existing SetWeight.
  void Clear() {
    first_ = kSetEmpty;
    rest_.clear();
  }

  size_t Size() const { return first_ == kSetEmpty ? 0 : rest_.size() + 1; }

  Label Back() {
    if (rest_.empty()) {
      return first_;
    } else {
      return rest_.back();
    }
  }

  // Caller must add in sort order and be unique (or error signalled).
  // Input should also be positive. Non-positive value (for the first
  // push) has special internal meaning w.r.t. integral constants above.
  void PushBack(Label label) {
    if (first_ == kSetEmpty) {
      first_ = label;
    } else {
      if (label <= Back() || label <= 0) {
        FSTERROR() << "SetWeight: labels must be positive, added"
                   << " in sort order and be unique.";
        rest_.push_back(Label(kSetBad));
      }
      rest_.push_back(label);
    }
  }

 private:
  Label first_ = kSetEmpty;  // First label in set (kSetEmpty if empty).
  std::list<Label> rest_;    // Remaining labels in set.
};

// Traverses set in forward direction.
template <class SetWeight_>
class SetWeightIterator {
 public:
  using Weight = SetWeight_;
  using Label = typename Weight::Label;

  explicit SetWeightIterator(const Weight &w)
      : first_(w.first_), rest_(w.rest_), init_(true), iter_(rest_.begin()) {}

  bool Done() const {
    if (init_) {
      return first_ == kSetEmpty;
    } else {
      return iter_ == rest_.end();
    }
  }

  const Label &Value() const { return init_ ? first_ : *iter_; }

  void Next() {
    if (init_) {
      init_ = false;
    } else {
      ++iter_;
    }
  }

  void Reset() {
    init_ = true;
    iter_ = rest_.begin();
  }

 private:
  const Label &first_;
  const decltype(Weight::rest_) &rest_;
  bool init_;  // In the initialized state?
  typename std::remove_reference<decltype(Weight::rest_)>::type::const_iterator iter_;
};


// SetWeight member functions follow that require SetWeightIterator

template <typename Label, SetType S>
inline std::istream &SetWeight<Label, S>::Read(std::istream &strm) {
  Clear();
  int32_t size;
  ReadType(strm, &size);
  for (int32_t i = 0; i < size; ++i) {
    Label label;
    ReadType(strm, &label);
    PushBack(label);
  }
  return strm;
}

template <typename Label, SetType S>
inline std::ostream &SetWeight<Label, S>::Write(std::ostream &strm) const {
  const int32_t size = Size();
  WriteType(strm, size);
  for (Iterator iter(*this); !iter.Done(); iter.Next()) {
    WriteType(strm, iter.Value());
  }
  return strm;
}

template <typename Label, SetType S>
inline bool SetWeight<Label, S>::Member() const {
  Iterator iter(*this);
  return iter.Value() != Label(kSetBad);
}

template <typename Label, SetType S>
inline typename SetWeight<Label, S>::ReverseWeight
SetWeight<Label, S>::Reverse() const {
  return *this;
}

template <typename Label, SetType S>
inline size_t SetWeight<Label, S>::Hash() const {
  using Weight = SetWeight<Label, S>;
  if (S == SET_BOOLEAN) {
    return *this == Weight::Zero() ? 0 : 1;
  } else {
    size_t h = 0;
    for (Iterator iter(*this); !iter.Done(); iter.Next()) {
      h ^= h << 1 ^ iter.Value();
    }
    return h;
  }
}

// Default ==
template <typename Label, SetType S>
inline bool operator==(const SetWeight<Label, S> &w1,
                       const SetWeight<Label, S> &w2) {
  if (w1.Size() != w2.Size()) return false;
  using Iterator = typename SetWeight<Label, S>::Iterator;
  Iterator iter1(w1);
  Iterator iter2(w2);
  for (; !iter1.Done(); iter1.Next(), iter2.Next()) {
    if (iter1.Value() != iter2.Value()) return false;
  }
  return true;
}

// Boolean ==
template <typename Label>
inline bool operator==(const SetWeight<Label, SET_BOOLEAN> &w1,
                       const SetWeight<Label, SET_BOOLEAN> &w2) {
  // x == kSetEmpty if x \nin {kUnivSet, kSetBad}
  if (!w1.Member() || !w2.Member()) return false;
  using Iterator = typename SetWeight<Label, SET_BOOLEAN>::Iterator;
  Iterator iter1(w1);
  Iterator iter2(w2);
  Label label1 = iter1.Done() ? kSetEmpty : iter1.Value();
  Label label2 = iter2.Done() ? kSetEmpty : iter2.Value();
  if (label1 == kSetUniv) return label2 == kSetUniv;
  if (label2 == kSetUniv) return label1 == kSetUniv;
  return true;
}

template <typename Label, SetType S>
inline bool operator!=(const SetWeight<Label, S> &w1,
                       const SetWeight<Label, S> &w2) {
  return !(w1 == w2);
}

template <typename Label, SetType S>
inline bool ApproxEqual(const SetWeight<Label, S> &w1,
                        const SetWeight<Label, S> &w2,
                        float delta = kDelta) {
  return w1 == w2;
}

template <typename Label, SetType S>
inline std::ostream &operator<<(std::ostream &strm,
                                const SetWeight<Label, S> &weight) {
  typename SetWeight<Label, S>::Iterator iter(weight);
  if (iter.Done()) {
    return strm << "EmptySet";
  } else if (iter.Value() == Label(kSetUniv)) {
    return strm << "UnivSet";
  } else if (iter.Value() == Label(kSetBad)) {
    return strm << "BadSet";
  } else {
    for (size_t i = 0; !iter.Done(); ++i, iter.Next()) {
      if (i > 0) strm << kSetSeparator;
      strm << iter.Value();
    }
  }
  return strm;
}

template <typename Label, SetType S>
inline std::istream &operator>>(std::istream &strm,
                                SetWeight<Label, S> &weight) {
  string str;
  strm >> str;
  using Weight = SetWeight<Label, S>;
  if (str == "EmptySet") {
    weight = Weight(Label(kSetEmpty));
  } else if (str == "UnivSet") {
    weight = Weight(Label(kSetUniv));
  } else {
    weight.Clear();
    char *p = nullptr;
    for (const char *cs = str.c_str(); !p || *p != '\0'; cs = p + 1) {
      const Label label = strtoll(cs, &p, 10);
      if (p == cs || (*p != 0 && *p != kSetSeparator)) {
        strm.clear(std::ios::badbit);
        break;
      }
      weight.PushBack(label);
    }
  }
  return strm;
}

template <typename Label, SetType S>
inline SetWeight<Label, S> Union(
    const SetWeight<Label, S> &w1,
    const SetWeight<Label, S> &w2) {
  using Weight = SetWeight<Label, S>;
  using Iterator = typename SetWeight<Label, S>::Iterator;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::EmptySet()) return w2;
  if (w2 == Weight::EmptySet()) return w1;
  if (w1 == Weight::UnivSet()) return w1;
  if (w2 == Weight::UnivSet()) return w2;
  Iterator it1(w1);
  Iterator it2(w2);
  Weight result;
  while (!it1.Done() && !it2.Done()) {
    const auto v1 = it1.Value();
    const auto v2 = it2.Value();
    if (v1 < v2) {
      result.PushBack(v1);
      it1.Next();
    } else if (v1 > v2) {
      result.PushBack(v2);
      it2.Next();
    } else {
      result.PushBack(v1);
      it1.Next();
      it2.Next();
    }
  }
  for (; !it1.Done(); it1.Next()) result.PushBack(it1.Value());
  for (; !it2.Done(); it2.Next()) result.PushBack(it2.Value());
  return result;
}

template <typename Label, SetType S>
inline SetWeight<Label, S> Intersect(
    const SetWeight<Label, S> &w1,
    const SetWeight<Label, S> &w2) {
  using Weight = SetWeight<Label, S>;
  using Iterator = typename SetWeight<Label, S>::Iterator;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::EmptySet()) return w1;
  if (w2 == Weight::EmptySet()) return w2;
  if (w1 == Weight::UnivSet()) return w2;
  if (w2 == Weight::UnivSet()) return w1;
  Iterator it1(w1);
  Iterator it2(w2);
  Weight result;
  while (!it1.Done() && !it2.Done()) {
    const auto v1 = it1.Value();
    const auto v2 = it2.Value();
    if (v1 < v2) {
      it1.Next();
    } else if (v1 > v2) {
      it2.Next();
    } else {
      result.PushBack(v1);
      it1.Next();
      it2.Next();
    }
  }
  return result;
}

template <typename Label, SetType S>
inline SetWeight<Label, S> Difference(
    const SetWeight<Label, S> &w1,
    const SetWeight<Label, S> &w2) {
  using Weight = SetWeight<Label, S>;
  using Iterator = typename SetWeight<Label, S>::Iterator;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::EmptySet()) return w1;
  if (w2 == Weight::EmptySet()) return w1;
  if (w2 == Weight::UnivSet()) return Weight::EmptySet();
  Iterator it1(w1);
  Iterator it2(w2);
  Weight result;
  while (!it1.Done() && !it2.Done()) {
    const auto v1 = it1.Value();
    const auto v2 = it2.Value();
    if (v1 < v2) {
      result.PushBack(v1);
      it1.Next();
    } else if (v1 > v2) {
      it2.Next();
    } else {
      it1.Next();
      it2.Next();
    }
  }
  for (; !it1.Done(); it1.Next()) result.PushBack(it1.Value());
  return result;
}

// Default: Plus = Intersect.
template <typename Label, SetType S>
inline SetWeight<Label, S> Plus(
    const SetWeight<Label, S> &w1,
    const SetWeight<Label, S> &w2) {
  return Intersect(w1, w2);
}

// Plus = Union.
template <typename Label>
inline SetWeight<Label, SET_UNION_INTERSECT> Plus(
    const SetWeight<Label, SET_UNION_INTERSECT> &w1,
    const SetWeight<Label, SET_UNION_INTERSECT> &w2) {
  return Union(w1, w2);
}

// Plus = Set equality is required (for non-Zero() input). The
// restriction is useful (e.g., in determinization) to ensure the input
// has a unique labelled path weight.
template <typename Label>
inline SetWeight<Label, SET_INTERSECT_UNION_RESTRICT> Plus(
    const SetWeight<Label, SET_INTERSECT_UNION_RESTRICT> &w1,
    const SetWeight<Label, SET_INTERSECT_UNION_RESTRICT> &w2) {
  using Weight = SetWeight<Label, SET_INTERSECT_UNION_RESTRICT>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::Zero()) return w2;
  if (w2 == Weight::Zero()) return w1;
  if (w1 != w2) {
    FSTERROR() << "SetWeight::Plus: Unequal arguments "
               << "(non-unique labelled path weights?)"
               << " w1 = " << w1 << " w2 = " << w2;
    return Weight::NoWeight();
  }
  return w1;
}

// Plus = Or.
template <typename Label>
inline SetWeight<Label, SET_BOOLEAN> Plus(
    const SetWeight<Label, SET_BOOLEAN> &w1,
    const SetWeight<Label, SET_BOOLEAN> &w2) {
  using Weight = SetWeight<Label, SET_BOOLEAN>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::One()) return w1;
  if (w2 == Weight::One()) return w2;
  return Weight::Zero();
}

// Default: Times = Union.
template <typename Label, SetType S>
inline SetWeight<Label, S> Times(
    const SetWeight<Label, S> &w1,
    const SetWeight<Label, S> &w2) {
  return Union(w1, w2);
}

// Times = Intersect.
template <typename Label>
inline SetWeight<Label, SET_UNION_INTERSECT> Times(
    const SetWeight<Label, SET_UNION_INTERSECT> &w1,
    const SetWeight<Label, SET_UNION_INTERSECT> &w2) {
  return Intersect(w1, w2);
}

// Times = And.
template <typename Label>
inline SetWeight<Label, SET_BOOLEAN> Times(
    const SetWeight<Label, SET_BOOLEAN> &w1,
    const SetWeight<Label, SET_BOOLEAN> &w2) {
  using Weight = SetWeight<Label, SET_BOOLEAN>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::One()) return w2;
  return w1;
}

// Divide = Difference.
template <typename Label, SetType S>
inline SetWeight<Label, S> Divide(const SetWeight<Label, S> &w1,
                                  const SetWeight<Label, S> &w2,
                                  DivideType divide_type = DIVIDE_ANY) {
  return Difference(w1, w2);
}

// Divide = dividend (or the universal set if the
// dividend == divisor).
template <typename Label>
inline SetWeight<Label, SET_UNION_INTERSECT> Divide(
    const SetWeight<Label, SET_UNION_INTERSECT> &w1,
    const SetWeight<Label, SET_UNION_INTERSECT> &w2,
    DivideType divide_type = DIVIDE_ANY) {
  using Weight = SetWeight<Label, SET_UNION_INTERSECT>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == w2) return Weight::UnivSet();
  return w1;
}

// Divide = Or Not.
template <typename Label>
inline SetWeight<Label, SET_BOOLEAN> Divide(
    const SetWeight<Label, SET_BOOLEAN> &w1,
    const SetWeight<Label, SET_BOOLEAN> &w2,
    DivideType divide_type = DIVIDE_ANY) {
  using Weight = SetWeight<Label, SET_BOOLEAN>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::One()) return w1;
  if (w2 == Weight::Zero()) return Weight::One();
  return Weight::Zero();
}

// Converts between different set types.
template <typename Label, SetType S1, SetType S2>
struct WeightConvert<SetWeight<Label, S1>, SetWeight<Label, S2>> {
  SetWeight<Label, S2> operator()(const SetWeight<Label, S1> &w1) const {
    using Iterator = SetWeightIterator<SetWeight<Label, S1>>;
    SetWeight<Label, S2> w2;
    for (Iterator iter(w1); !iter.Done(); iter.Next())
      w2.PushBack(iter.Value());
    return w2;
  }
};

// This function object generates SetWeights that are random integer sets
// from {1, ... , alphabet_size}^{0, max_set_length} U { Zero }. This is
// intended primarily for testing.
template <class Label, SetType S>
class WeightGenerate<SetWeight<Label, S>> {
 public:
  using Weight = SetWeight<Label, S>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t alphabet_size = kNumRandomWeights,
                          size_t max_set_length = kNumRandomWeights)
      : allow_zero_(allow_zero),
        alphabet_size_(alphabet_size),
        max_set_length_(max_set_length) {}

  Weight operator()() const {
    const size_t n = rand() % (max_set_length_ + allow_zero_);  // NOLINT
    if (allow_zero_ && n == max_set_length_) return Weight::Zero();
    std::vector<Label> labels;
    for (size_t i = 0; i < n; ++i) {
      labels.push_back(rand() % alphabet_size_ + 1);  // NOLINT
    }
    std::sort(labels.begin(), labels.end());
    const auto labels_end = std::unique(labels.begin(), labels.end());
    labels.resize(labels_end - labels.begin());
    return Weight(labels.begin(), labels.end());
  }

 private:
  // Permits Zero() and zero divisors.
  const bool allow_zero_;
  // Alphabet size for random weights.
  const size_t alphabet_size_;
  // Number of alternative random weights.
  const size_t max_set_length_;
};

}  // namespace fst

#endif  // FST_SET_WEIGHT_H_
