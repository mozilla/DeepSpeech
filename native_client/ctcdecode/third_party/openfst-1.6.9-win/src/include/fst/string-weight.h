// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// String weight set and associated semiring operation definitions.

#ifndef FST_STRING_WEIGHT_H_
#define FST_STRING_WEIGHT_H_

#include <cstdlib>

#include <list>
#include <string>
#include <vector>

#include <fst/product-weight.h>
#include <fst/union-weight.h>
#include <fst/weight.h>


namespace fst {

constexpr int kStringInfinity = -1;     // Label for the infinite string.
constexpr int kStringBad = -2;          // Label for a non-string.
constexpr char kStringSeparator = '_';  // Label separator in strings.

// Determines whether to use left or right string semiring.  Includes a
// 'restricted' version that signals an error if proper prefixes/suffixes
// would otherwise be returned by Plus, useful with various
// algorithms that require functional transducer input with the
// string semirings.
enum StringType { STRING_LEFT = 0, STRING_RIGHT = 1, STRING_RESTRICT = 2 };

constexpr StringType ReverseStringType(StringType s) {
  return s == STRING_LEFT ? STRING_RIGHT
                          : (s == STRING_RIGHT ? STRING_LEFT : STRING_RESTRICT);
}

template <class>
class StringWeightIterator;
template <class>
class StringWeightReverseIterator;

// String semiring: (longest_common_prefix/suffix, ., Infinity, Epsilon)
template <typename Label_, StringType S = STRING_LEFT>
class StringWeight {
 public:
  using Label = Label_;
  using ReverseWeight = StringWeight<Label, ReverseStringType(S)>;
  using Iterator = StringWeightIterator<StringWeight>;
  using ReverseIterator = StringWeightReverseIterator<StringWeight>;

  friend class StringWeightIterator<StringWeight>;
  friend class StringWeightReverseIterator<StringWeight>;

  StringWeight() {}

  template <typename Iterator>
  StringWeight(const Iterator &begin, const Iterator &end) {
    for (auto iter = begin; iter != end; ++iter) PushBack(*iter);
  }

  explicit StringWeight(Label label) { PushBack(label); }

  static const StringWeight &Zero() {
    static const auto *const zero = new StringWeight(Label(kStringInfinity));
    return *zero;
  }

  static const StringWeight &One() {
    static const auto *const one = new StringWeight();
    return *one;
  }

  static const StringWeight &NoWeight() {
    static const auto *const no_weight = new StringWeight(Label(kStringBad));
    return *no_weight;
  }

  static const string &Type() {
    static const string *const type = new string(
        S == STRING_LEFT
            ? "left_string"
            : (S == STRING_RIGHT ? "right_string" : "restricted_string"));
    return *type;
  }

  bool Member() const;

  std::istream &Read(std::istream &strm);

  std::ostream &Write(std::ostream &strm) const;

  size_t Hash() const;

  StringWeight Quantize(float delta = kDelta) const { return *this; }

  ReverseWeight Reverse() const;

  static constexpr uint64_t Properties() {
    return kIdempotent |
           (S == STRING_LEFT ? kLeftSemiring
                             : (S == STRING_RIGHT
                                    ? kRightSemiring
                                    : /* S == STRING_RESTRICT */ kLeftSemiring |
                                          kRightSemiring));
  }

  // These operations combined with the StringWeightIterator and
  // StringWeightReverseIterator provide the access and mutation of the string
  // internal elements.

  // Clear existing StringWeight.
  void Clear() {
    first_ = 0;
    rest_.clear();
  }

  size_t Size() const { return first_ ? rest_.size() + 1 : 0; }

  void PushFront(Label label) {
    if (first_) rest_.push_front(first_);
    first_ = label;
  }

  void PushBack(Label label) {
    if (!first_) {
      first_ = label;
    } else {
      rest_.push_back(label);
    }
  }

 private:
  Label first_ = 0;        // First label in string (0 if empty).
  std::list<Label> rest_;  // Remaining labels in string.
};

// Traverses string in forward direction.
template <class StringWeight_>
class StringWeightIterator {
 public:
  using Weight = StringWeight_;
  using Label = typename Weight::Label;

  explicit StringWeightIterator(const Weight &w)
      : first_(w.first_), rest_(w.rest_), init_(true), iter_(rest_.begin()) {}

  bool Done() const {
    if (init_) {
      return first_ == 0;
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
  typename std::remove_reference<decltype (Weight::rest_)>::type::const_iterator iter_;
};

// Traverses string in backward direction.
template <class StringWeight_>
class StringWeightReverseIterator {
 public:
  using Weight = StringWeight_;
  using Label = typename Weight::Label;

  explicit StringWeightReverseIterator(const Weight &w)
      : first_(w.first_),
        rest_(w.rest_),
        fin_(first_ == Label()),
        iter_(rest_.rbegin()) {}

  bool Done() const { return fin_; }

  const Label &Value() const { return iter_ == rest_.rend() ? first_ : *iter_; }

  void Next() {
    if (iter_ == rest_.rend()) {
      fin_ = true;
    } else {
      ++iter_;
    }
  }

  void Reset() {
    fin_ = false;
    iter_ = rest_.rbegin();
  }

 private:
  const Label &first_;
  const decltype(Weight::rest_) &rest_;
  bool fin_;  // In the final state?
  typename std::remove_reference<decltype (Weight::rest_)>::type::const_reverse_iterator iter_;
};

// StringWeight member functions follow that require
// StringWeightIterator or StringWeightReverseIterator.

template <typename Label, StringType S>
inline std::istream &StringWeight<Label, S>::Read(std::istream &strm) {
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

template <typename Label, StringType S>
inline std::ostream &StringWeight<Label, S>::Write(std::ostream &strm) const {
  const int32_t size = Size();
  WriteType(strm, size);
  for (Iterator iter(*this); !iter.Done(); iter.Next()) {
    WriteType(strm, iter.Value());
  }
  return strm;
}

template <typename Label, StringType S>
inline bool StringWeight<Label, S>::Member() const {
  Iterator iter(*this);
  return iter.Value() != Label(kStringBad);
}

template <typename Label, StringType S>
inline typename StringWeight<Label, S>::ReverseWeight
StringWeight<Label, S>::Reverse() const {
  ReverseWeight rweight;
  for (Iterator iter(*this); !iter.Done(); iter.Next()) {
    rweight.PushFront(iter.Value());
  }
  return rweight;
}

template <typename Label, StringType S>
inline size_t StringWeight<Label, S>::Hash() const {
  size_t h = 0;
  for (Iterator iter(*this); !iter.Done(); iter.Next()) {
    h ^= h << 1 ^ iter.Value();
  }
  return h;
}

template <typename Label, StringType S>
inline bool operator==(const StringWeight<Label, S> &w1,
                       const StringWeight<Label, S> &w2) {
  if (w1.Size() != w2.Size()) return false;
  using Iterator = typename StringWeight<Label, S>::Iterator;
  Iterator iter1(w1);
  Iterator iter2(w2);
  for (; !iter1.Done(); iter1.Next(), iter2.Next()) {
    if (iter1.Value() != iter2.Value()) return false;
  }
  return true;
}

template <typename Label, StringType S>
inline bool operator!=(const StringWeight<Label, S> &w1,
                       const StringWeight<Label, S> &w2) {
  return !(w1 == w2);
}

template <typename Label, StringType S>
inline bool ApproxEqual(const StringWeight<Label, S> &w1,
                        const StringWeight<Label, S> &w2,
                        float delta = kDelta) {
  return w1 == w2;
}

template <typename Label, StringType S>
inline std::ostream &operator<<(std::ostream &strm,
                                const StringWeight<Label, S> &weight) {
  typename StringWeight<Label, S>::Iterator iter(weight);
  if (iter.Done()) {
    return strm << "Epsilon";
  } else if (iter.Value() == Label(kStringInfinity)) {
    return strm << "Infinity";
  } else if (iter.Value() == Label(kStringBad)) {
    return strm << "BadString";
  } else {
    for (size_t i = 0; !iter.Done(); ++i, iter.Next()) {
      if (i > 0) strm << kStringSeparator;
      strm << iter.Value();
    }
  }
  return strm;
}

template <typename Label, StringType S>
inline std::istream &operator>>(std::istream &strm,
                                StringWeight<Label, S> &weight) {
  string str;
  strm >> str;
  using Weight = StringWeight<Label, S>;
  if (str == "Infinity") {
    weight = Weight::Zero();
  } else if (str == "Epsilon") {
    weight = Weight::One();
  } else {
    weight.Clear();
    char *p = nullptr;
    for (const char *cs = str.c_str(); !p || *p != '\0'; cs = p + 1) {
      const Label label = strtoll(cs, &p, 10);
      if (p == cs || (*p != 0 && *p != kStringSeparator)) {
        strm.clear(std::ios::badbit);
        break;
      }
      weight.PushBack(label);
    }
  }
  return strm;
}

// Default is for the restricted string semiring. String equality is required
// (for non-Zero() input). The restriction is used (e.g., in determinization)
// to ensure the input is functional.
template <typename Label, StringType S>
inline StringWeight<Label, S> Plus(const StringWeight<Label, S> &w1,
                                   const StringWeight<Label, S> &w2) {
  using Weight = StringWeight<Label, S>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::Zero()) return w2;
  if (w2 == Weight::Zero()) return w1;
  if (w1 != w2) {
    FSTERROR() << "StringWeight::Plus: Unequal arguments "
               << "(non-functional FST?)"
               << " w1 = " << w1 << " w2 = " << w2;
    return Weight::NoWeight();
  }
  return w1;
}

// Longest common prefix for left string semiring.
template <typename Label>
inline StringWeight<Label, STRING_LEFT> Plus(
    const StringWeight<Label, STRING_LEFT> &w1,
    const StringWeight<Label, STRING_LEFT> &w2) {
  using Weight = StringWeight<Label, STRING_LEFT>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::Zero()) return w2;
  if (w2 == Weight::Zero()) return w1;
  Weight sum;
  typename Weight::Iterator iter1(w1);
  typename Weight::Iterator iter2(w2);
  for (; !iter1.Done() && !iter2.Done() && iter1.Value() == iter2.Value();
       iter1.Next(), iter2.Next()) {
    sum.PushBack(iter1.Value());
  }
  return sum;
}

// Longest common suffix for right string semiring.
template <typename Label>
inline StringWeight<Label, STRING_RIGHT> Plus(
    const StringWeight<Label, STRING_RIGHT> &w1,
    const StringWeight<Label, STRING_RIGHT> &w2) {
  using Weight = StringWeight<Label, STRING_RIGHT>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::Zero()) return w2;
  if (w2 == Weight::Zero()) return w1;
  Weight sum;
  typename Weight::ReverseIterator iter1(w1);
  typename Weight::ReverseIterator iter2(w2);
  for (; !iter1.Done() && !iter2.Done() && iter1.Value() == iter2.Value();
       iter1.Next(), iter2.Next()) {
    sum.PushFront(iter1.Value());
  }
  return sum;
}

template <typename Label, StringType S>
inline StringWeight<Label, S> Times(const StringWeight<Label, S> &w1,
                                    const StringWeight<Label, S> &w2) {
  using Weight = StringWeight<Label, S>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w1 == Weight::Zero() || w2 == Weight::Zero()) return Weight::Zero();
  Weight product(w1);
  for (typename Weight::Iterator iter(w2); !iter.Done(); iter.Next()) {
    product.PushBack(iter.Value());
  }
  return product;
}

// Left division in a left string semiring.
template <typename Label, StringType S>
inline StringWeight<Label, S> DivideLeft(const StringWeight<Label, S> &w1,
                                         const StringWeight<Label, S> &w2) {
  using Weight = StringWeight<Label, S>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w2 == Weight::Zero()) {
    return Weight(Label(kStringBad));
  } else if (w1 == Weight::Zero()) {
    return Weight::Zero();
  }
  Weight result;
  typename Weight::Iterator iter(w1);
  size_t i = 0;
  for (; !iter.Done() && i < w2.Size(); iter.Next(), ++i) {
  }
  for (; !iter.Done(); iter.Next()) result.PushBack(iter.Value());
  return result;
}

// Right division in a right string semiring.
template <typename Label, StringType S>
inline StringWeight<Label, S> DivideRight(const StringWeight<Label, S> &w1,
                                          const StringWeight<Label, S> &w2) {
  using Weight = StringWeight<Label, S>;
  if (!w1.Member() || !w2.Member()) return Weight::NoWeight();
  if (w2 == Weight::Zero()) {
    return Weight(Label(kStringBad));
  } else if (w1 == Weight::Zero()) {
    return Weight::Zero();
  }
  Weight result;
  typename Weight::ReverseIterator iter(w1);
  size_t i = 0;
  for (; !iter.Done() && i < w2.Size(); iter.Next(), ++i) {
  }
  for (; !iter.Done(); iter.Next()) result.PushFront(iter.Value());
  return result;
}

// Default is the restricted string semiring.
template <typename Label, StringType S>
inline StringWeight<Label, S> Divide(const StringWeight<Label, S> &w1,
                                     const StringWeight<Label, S> &w2,
                                     DivideType divide_type) {
  using Weight = StringWeight<Label, S>;
  if (divide_type == DIVIDE_LEFT) {
    return DivideLeft(w1, w2);
  } else if (divide_type == DIVIDE_RIGHT) {
    return DivideRight(w1, w2);
  } else {
    FSTERROR() << "StringWeight::Divide: "
               << "Only explicit left or right division is defined "
               << "for the " << Weight::Type() << " semiring";
    return Weight::NoWeight();
  }
}

// Left division in the left string semiring.
template <typename Label>
inline StringWeight<Label, STRING_LEFT> Divide(
    const StringWeight<Label, STRING_LEFT> &w1,
    const StringWeight<Label, STRING_LEFT> &w2, DivideType divide_type) {
  if (divide_type != DIVIDE_LEFT) {
    FSTERROR() << "StringWeight::Divide: Only left division is defined "
               << "for the left string semiring";
    return StringWeight<Label, STRING_LEFT>::NoWeight();
  }
  return DivideLeft(w1, w2);
}

// Right division in the right string semiring.
template <typename Label>
inline StringWeight<Label, STRING_RIGHT> Divide(
    const StringWeight<Label, STRING_RIGHT> &w1,
    const StringWeight<Label, STRING_RIGHT> &w2, DivideType divide_type) {
  if (divide_type != DIVIDE_RIGHT) {
    FSTERROR() << "StringWeight::Divide: Only right division is defined "
               << "for the right string semiring";
    return StringWeight<Label, STRING_RIGHT>::NoWeight();
  }
  return DivideRight(w1, w2);
}

// This function object generates StringWeights that are random integer strings
// from {1, ... , alphabet_size)^{0, max_string_length} U { Zero }. This is
// intended primarily for testing.
template <class Label, StringType S>
class WeightGenerate<StringWeight<Label, S>> {
 public:
  using Weight = StringWeight<Label, S>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t alphabet_size = kNumRandomWeights,
                          size_t max_string_length = kNumRandomWeights)
      : allow_zero_(allow_zero),
        alphabet_size_(alphabet_size),
        max_string_length_(max_string_length) {}

  Weight operator()() const {
    size_t n = rand() % (max_string_length_ + allow_zero_);  // NOLINT
    if (allow_zero_ && n == max_string_length_) return Weight::Zero();
    std::vector<Label> labels;
    labels.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      labels.push_back(rand() % alphabet_size_ + 1);  // NOLINT
    }
    return Weight(labels.begin(), labels.end());
  }

 private:
  // Permits Zero() and zero divisors.
  const bool allow_zero_;
  // Alphabet size for random weights.
  const size_t alphabet_size_;
  // Number of alternative random weights.
  const size_t max_string_length_;
};

// Determines whether to use left, right, or (general) gallic semiring. Includes
// a restricted version that signals an error if proper string prefixes or
// suffixes would otherwise be returned by string Plus. This is useful with
// algorithms that require functional transducer input. Also includes min
// version that changes the Plus to keep only the lowest W weight string.
enum GallicType {
  GALLIC_LEFT = 0,
  GALLIC_RIGHT = 1,
  GALLIC_RESTRICT = 2,
  GALLIC_MIN = 3,
  GALLIC = 4
};

constexpr StringType GallicStringType(GallicType g) {
  return g == GALLIC_LEFT
             ? STRING_LEFT
             : (g == GALLIC_RIGHT ? STRING_RIGHT : STRING_RESTRICT);
}

constexpr GallicType ReverseGallicType(GallicType g) {
  return g == GALLIC_LEFT
             ? GALLIC_RIGHT
             : (g == GALLIC_RIGHT
                    ? GALLIC_LEFT
                    : (g == GALLIC_RESTRICT
                           ? GALLIC_RESTRICT
                           : (g == GALLIC_MIN ? GALLIC_MIN : GALLIC)));
}

// Product of string weight and an arbitraryy weight.
template <class Label, class W, GallicType G = GALLIC_LEFT>
struct GallicWeight
    : public ProductWeight<StringWeight<Label, GallicStringType(G)>, W> {
  using ReverseWeight =
      GallicWeight<Label, typename W::ReverseWeight, ReverseGallicType(G)>;
  using SW = StringWeight<Label, GallicStringType(G)>;

  using ProductWeight<SW, W>::Properties;

  GallicWeight() {}

  GallicWeight(SW w1, W w2) : ProductWeight<SW, W>(w1, w2) {}

  explicit GallicWeight(const string &s, int *nread = nullptr)
      : ProductWeight<SW, W>(s, nread) {}

  explicit GallicWeight(const ProductWeight<SW, W> &w)
      : ProductWeight<SW, W>(w) {}

  static const GallicWeight &Zero() {
    static const GallicWeight zero(ProductWeight<SW, W>::Zero());
    return zero;
  }

  static const GallicWeight &One() {
    static const GallicWeight one(ProductWeight<SW, W>::One());
    return one;
  }

  static const GallicWeight &NoWeight() {
    static const GallicWeight no_weight(ProductWeight<SW, W>::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type = new string(
        G == GALLIC_LEFT
            ? "left_gallic"
            : (G == GALLIC_RIGHT
                   ? "right_gallic"
                   : (G == GALLIC_RESTRICT
                          ? "restricted_gallic"
                          : (G == GALLIC_MIN ? "min_gallic" : "gallic"))));
    return *type;
  }

  GallicWeight Quantize(float delta = kDelta) const {
    return GallicWeight(ProductWeight<SW, W>::Quantize(delta));
  }

  ReverseWeight Reverse() const {
    return ReverseWeight(ProductWeight<SW, W>::Reverse());
  }
};

// Default plus.
template <class Label, class W, GallicType G>
inline GallicWeight<Label, W, G> Plus(const GallicWeight<Label, W, G> &w,
                                      const GallicWeight<Label, W, G> &v) {
  return GallicWeight<Label, W, G>(Plus(w.Value1(), v.Value1()),
                                   Plus(w.Value2(), v.Value2()));
}

// Min gallic plus.
template <class Label, class W>
inline GallicWeight<Label, W, GALLIC_MIN> Plus(
    const GallicWeight<Label, W, GALLIC_MIN> &w1,
    const GallicWeight<Label, W, GALLIC_MIN> &w2) {
  static const NaturalLess<W> less;
  return less(w1.Value2(), w2.Value2()) ? w1 : w2;
}

template <class Label, class W, GallicType G>
inline GallicWeight<Label, W, G> Times(const GallicWeight<Label, W, G> &w,
                                       const GallicWeight<Label, W, G> &v) {
  return GallicWeight<Label, W, G>(Times(w.Value1(), v.Value1()),
                                   Times(w.Value2(), v.Value2()));
}

template <class Label, class W, GallicType G>
inline GallicWeight<Label, W, G> Divide(const GallicWeight<Label, W, G> &w,
                                        const GallicWeight<Label, W, G> &v,
                                        DivideType divide_type = DIVIDE_ANY) {
  return GallicWeight<Label, W, G>(Divide(w.Value1(), v.Value1(), divide_type),
                                   Divide(w.Value2(), v.Value2(), divide_type));
}

// This function object generates gallic weights by calling an underlying
// product weight generator. This is intended primarily for testing.
template <class Label, class W, GallicType G>
class WeightGenerate<GallicWeight<Label, W, G>>
    : public WeightGenerate<
          ProductWeight<StringWeight<Label, GallicStringType(G)>, W>> {
 public:
  using Weight = GallicWeight<Label, W, G>;
  using Generate = WeightGenerate<
      ProductWeight<StringWeight<Label, GallicStringType(G)>, W>>;

  explicit WeightGenerate(bool allow_zero = true) : generate_(allow_zero) {}

  Weight operator()() const { return Weight(generate_()); }

 private:
  const Generate generate_;
};

// Union weight options for (general) GALLIC type.
template <class Label, class W>
struct GallicUnionWeightOptions {
  using ReverseOptions = GallicUnionWeightOptions<Label, W>;
  using GW = GallicWeight<Label, W, GALLIC_RESTRICT>;
  using SW = StringWeight<Label, GallicStringType(GALLIC_RESTRICT)>;
  using SI = StringWeightIterator<SW>;

  // Military order.
  struct Compare {
    bool operator()(const GW &w1, const GW &w2) const {
      const SW &s1 = w1.Value1();
      const SW &s2 = w2.Value1();
      if (s1.Size() < s2.Size()) return true;
      if (s1.Size() > s2.Size()) return false;
      SI iter1(s1);
      SI iter2(s2);
      while (!iter1.Done()) {
        const auto l1 = iter1.Value();
        const auto l2 = iter2.Value();
        if (l1 < l2) return true;
        if (l1 > l2) return false;
        iter1.Next();
        iter2.Next();
      }
      return false;
    }
  };

  // Adds W weights when string part equal.
  struct Merge {
    GW operator()(const GW &w1, const GW &w2) const {
      return GW(w1.Value1(), Plus(w1.Value2(), w2.Value2()));
    }
  };
};

// Specialization for the (general) GALLIC type.
template <class Label, class W>
struct GallicWeight<Label, W, GALLIC>
    : public UnionWeight<GallicWeight<Label, W, GALLIC_RESTRICT>,
                         GallicUnionWeightOptions<Label, W>> {
  using GW = GallicWeight<Label, W, GALLIC_RESTRICT>;
  using SW = StringWeight<Label, GallicStringType(GALLIC_RESTRICT)>;
  using SI = StringWeightIterator<SW>;
  using UW = UnionWeight<GW, GallicUnionWeightOptions<Label, W>>;
  using UI = UnionWeightIterator<GW, GallicUnionWeightOptions<Label, W>>;
  using ReverseWeight = GallicWeight<Label, W, GALLIC>;

  using UW::Properties;

  GallicWeight() {}

  // Copy constructor.
  GallicWeight(const UW &weight) : UW(weight) {}  // NOLINT

  // Singleton constructors: create a GALLIC weight containing a single
  // GALLIC_RESTRICT weight. Takes as argument (1) a GALLIC_RESTRICT weight or
  // (2) the two components of a GALLIC_RESTRICT weight.
  explicit GallicWeight(const GW &weight) : UW(weight) {}

  GallicWeight(SW w1, W w2) : UW(GW(w1, w2)) {}

  explicit GallicWeight(const string &str, int *nread = nullptr)
      : UW(str, nread) {}

  static const GallicWeight<Label, W, GALLIC> &Zero() {
    static const GallicWeight<Label, W, GALLIC> zero(UW::Zero());
    return zero;
  }

  static const GallicWeight<Label, W, GALLIC> &One() {
    static const GallicWeight<Label, W, GALLIC> one(UW::One());
    return one;
  }

  static const GallicWeight<Label, W, GALLIC> &NoWeight() {
    static const GallicWeight<Label, W, GALLIC> no_weight(UW::NoWeight());
    return no_weight;
  }

  static const string &Type() {
    static const string *const type = new string("gallic");
    return *type;
  }

  GallicWeight<Label, W, GALLIC> Quantize(float delta = kDelta) const {
    return UW::Quantize(delta);
  }

  ReverseWeight Reverse() const { return UW::Reverse(); }
};

// (General) gallic plus.
template <class Label, class W>
inline GallicWeight<Label, W, GALLIC> Plus(
    const GallicWeight<Label, W, GALLIC> &w1,
    const GallicWeight<Label, W, GALLIC> &w2) {
  using GW = GallicWeight<Label, W, GALLIC_RESTRICT>;
  using UW = UnionWeight<GW, GallicUnionWeightOptions<Label, W>>;
  return Plus(static_cast<UW>(w1), static_cast<UW>(w2));
}

// (General) gallic times.
template <class Label, class W>
inline GallicWeight<Label, W, GALLIC> Times(
    const GallicWeight<Label, W, GALLIC> &w1,
    const GallicWeight<Label, W, GALLIC> &w2) {
  using GW = GallicWeight<Label, W, GALLIC_RESTRICT>;
  using UW = UnionWeight<GW, GallicUnionWeightOptions<Label, W>>;
  return Times(static_cast<UW>(w1), static_cast<UW>(w2));
}

// (General) gallic divide.
template <class Label, class W>
inline GallicWeight<Label, W, GALLIC> Divide(
    const GallicWeight<Label, W, GALLIC> &w1,
    const GallicWeight<Label, W, GALLIC> &w2,
    DivideType divide_type = DIVIDE_ANY) {
  using GW = GallicWeight<Label, W, GALLIC_RESTRICT>;
  using UW = UnionWeight<GW, GallicUnionWeightOptions<Label, W>>;
  return Divide(static_cast<UW>(w1), static_cast<UW>(w2), divide_type);
}

// This function object generates gallic weights by calling an underlying
// union weight generator. This is intended primarily for testing.
template <class Label, class W>
class WeightGenerate<GallicWeight<Label, W, GALLIC>>
    : public WeightGenerate<UnionWeight<GallicWeight<Label, W, GALLIC_RESTRICT>,
                                        GallicUnionWeightOptions<Label, W>>> {
 public:
  using Weight = GallicWeight<Label, W, GALLIC>;
  using Generate =
      WeightGenerate<UnionWeight<GallicWeight<Label, W, GALLIC_RESTRICT>,
                                 GallicUnionWeightOptions<Label, W>>>;

  explicit WeightGenerate(bool allow_zero = true) : generate_(allow_zero) {}

  Weight operator()() const { return Weight(generate_()); }

 private:
  const Generate generate_;
};

}  // namespace fst

#endif  // FST_STRING_WEIGHT_H_
