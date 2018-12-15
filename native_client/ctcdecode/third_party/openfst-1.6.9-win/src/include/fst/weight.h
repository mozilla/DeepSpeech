// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// General weight set and associated semiring operation definitions.

#ifndef FST_WEIGHT_H_
#define FST_WEIGHT_H_

#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <utility>

#include <fst/compat.h>
#include <fst/log.h>

#include <fst/util.h>


DECLARE_string(fst_weight_parentheses);
DECLARE_string(fst_weight_separator);

namespace fst {

// A semiring is specified by two binary operations Plus and Times and two
// designated elements Zero and One with the following properties:
//
//   Plus: associative, commutative, and has Zero as its identity.
//
//   Times: associative and has identity One, distributes w.r.t. Plus, and
//     has Zero as an annihilator:
//          Times(Zero(), a) == Times(a, Zero()) = Zero().
//
// A left semiring distributes on the left; a right semiring is similarly
// defined.
//
// A Weight class must have binary functions Plus and Times and static member
// functions Zero() and One() and these must form (at least) a left or right
// semiring.
//
// In addition, the following should be defined for a Weight:
//
//   Member: predicate on set membership.
//
//   NoWeight: static member function that returns an element that is
//      not a set member; used to signal an error.
//
//   >>: reads textual representation of a weight.
//
//   <<: prints textual representation of a weight.
//
//   Read(istream &istrm): reads binary representation of a weight.
//
//   Write(ostream &ostrm): writes binary representation of a weight.
//
//   Hash: maps weight to size_t.
//
//   ApproxEqual: approximate equality (for inexact weights)
//
//   Quantize: quantizes w.r.t delta (for inexact weights)
//
//   Divide: for all a, b, c s.t. Times(a, b) == c
//
//     --> b' = Divide(c, a, DIVIDE_LEFT) if a left semiring, b'.Member()
//      and Times(a, b') == c
//     --> a' = Divide(c, b, DIVIDE_RIGHT) if a right semiring, a'.Member()
//      and Times(a', b) == c
//     --> b' = Divide(c, a) = Divide(c, a, DIVIDE_ANY) =
//      Divide(c, a, DIVIDE_LEFT) = Divide(c, a, DIVIDE_RIGHT) if a
//      commutative semiring, b'.Member() and Times(a, b') = Times(b', a) = c
//
//   ReverseWeight: the type of the corresponding reverse weight.
//
//     Typically the same type as Weight for a (both left and right) semiring.
//     For the left string semiring, it is the right string semiring.
//
//   Reverse: a mapping from Weight to ReverseWeight s.t.
//
//     --> Reverse(Reverse(a)) = a
//     --> Reverse(Plus(a, b)) = Plus(Reverse(a), Reverse(b))
//     --> Reverse(Times(a, b)) = Times(Reverse(b), Reverse(a))
//     Typically the identity mapping in a (both left and right) semiring.
//     In the left string semiring, it maps to the reverse string in the right
//     string semiring.
//
//   Properties: specifies additional properties that hold:
//      LeftSemiring: indicates weights form a left semiring.
//      RightSemiring: indicates weights form a right semiring.
//      Commutative: for all a,b: Times(a,b) == Times(b,a)
//      Idempotent: for all a: Plus(a, a) == a.
//      Path: for all a, b: Plus(a, b) == a or Plus(a, b) == b.

// CONSTANT DEFINITIONS

// A representable float near .001.
constexpr float kDelta = 1.0F / 1024.0F;

// For all a, b, c: Times(c, Plus(a, b)) = Plus(Times(c, a), Times(c, b)).
constexpr uint64_t kLeftSemiring = 0x0000000000000001ULL;

// For all a, b, c: Times(Plus(a, b), c) = Plus(Times(a, c), Times(b, c)).
constexpr uint64_t kRightSemiring = 0x0000000000000002ULL;

constexpr uint64_t kSemiring = kLeftSemiring | kRightSemiring;

// For all a, b: Times(a, b) = Times(b, a).
constexpr uint64_t kCommutative = 0x0000000000000004ULL;

// For all a: Plus(a, a) = a.
constexpr uint64_t kIdempotent = 0x0000000000000008ULL;

// For all a, b: Plus(a, b) = a or Plus(a, b) = b.
constexpr uint64_t kPath = 0x0000000000000010ULL;

// For random weight generation: default number of distinct weights.
// This is also used for a few other weight generation defaults.
constexpr size_t kNumRandomWeights = 5;

// Weight property boolean constants needed for SFINAE.

// MSVC compiler bug workaround: an expression containing W::Properties() cannot
// be directly used as a value argument to std::enable_if or integral_constant.
// WeightPropertiesThunk<W>::Properties works instead, however.
namespace bug {
template <class W>
struct WeightPropertiesThunk {
  WeightPropertiesThunk() = delete;
  constexpr static const uint64_t Properties = W::Properties();
};

template <class W, uint64_t props>
using TestWeightProperties = std::integral_constant<bool,
        (WeightPropertiesThunk<W>::Properties & props) == props>;
}  // namespace bug

template <class W>
using IsIdempotent = bug::TestWeightProperties<W, kIdempotent>;

template <class W>
using IsPath = bug::TestWeightProperties<W, kPath>;


// Determines direction of division.
enum DivideType {
  DIVIDE_LEFT,   // left division
  DIVIDE_RIGHT,  // right division
  DIVIDE_ANY
};  // division in a commutative semiring

// NATURAL ORDER
//
// By definition:
//
//                 a <= b iff a + b = a
//
// The natural order is a negative partial order iff the semiring is
// idempotent. It is trivially monotonic for plus. It is left
// (resp. right) monotonic for times iff the semiring is left
// (resp. right) distributive. It is a total order iff the semiring
// has the path property.
//
// For more information, see:
//
// Mohri, M. 2002. Semiring framework and algorithms for shortest-distance
// problems, Journal of Automata, Languages and
// Combinatorics 7(3): 321-350, 2002.
//
// We define the strict version of this order below.

template <class W>
class NaturalLess {
public:
  using Weight = W;

  NaturalLess() {
    if (!(W::Properties() & kIdempotent)) {
      FSTERROR() << "NaturalLess: Weight type is not idempotent: " << W::Type();
    }
  }

  bool operator()(const W &w1, const W &w2) const {
    return (Plus(w1, w2) == w1) && w1 != w2;
  }
};

// Power is the iterated product for arbitrary semirings such that Power(w, 0)
// is One() for the semiring, and Power(w, n) = Times(Power(w, n - 1), w).
template <class Weight>
Weight Power(const Weight &weight, size_t n) {
  auto result = Weight::One();
  for (size_t i = 0; i < n; ++i) result = Times(result, weight);
  return result;
}

// Simple default adder class. Specializations might be more complex.
template <class Weight>
class Adder {
 public:
  explicit Adder(Weight w = Weight::Zero()) : sum_(w) { }

  Weight Add(const Weight &w) {
    sum_ = Plus(sum_, w);
    return sum_;
  }

  Weight Sum() { return sum_; }

  void Reset(Weight w = Weight::Zero()) { sum_ = w; }

 private:
  Weight sum_;
};

// General weight converter: raises error.
template <class W1, class W2>
struct WeightConvert {
  W2 operator()(W1 w1) const {
    FSTERROR() << "WeightConvert: Can't convert weight from \"" << W1::Type()
               << "\" to \"" << W2::Type();
    return W2::NoWeight();
  }
};

// Specialized weight converter to self.
template <class W>
struct WeightConvert<W, W> {
  W operator()(W weight) const { return weight; }
};

// General random weight generator: raises error.
template <class W>
struct WeightGenerate {
  W operator()() const {
    FSTERROR() << "WeightGenerate: No random generator for " << W::Type();
    return W::NoWeight();
  }
};

namespace internal {

class CompositeWeightIO {
 public:
  CompositeWeightIO();
  CompositeWeightIO(char separator, std::pair<char, char> parentheses);

  std::pair<char, char> parentheses() const {
    return {open_paren_, close_paren_};
  }
  char separator() const { return separator_; }

  bool error() const { return error_; }

 protected:
  const char separator_;
  const char open_paren_;
  const char close_paren_;

 private:
  bool error_;
};

}  // namespace internal

// Helper class for writing textual composite weights.
class CompositeWeightWriter : public internal::CompositeWeightIO {
 public:
  // Uses configuration from flags (FLAGS_fst_weight_separator,
  // FLAGS_fst_weight_parentheses).
  explicit CompositeWeightWriter(std::ostream &ostrm);

  // parentheses defines the opening and closing parenthesis characters.
  // Set parentheses = {0, 0} to disable writing parenthesis.
  CompositeWeightWriter(std::ostream &ostrm, char separator,
                        std::pair<char, char> parentheses);

  CompositeWeightWriter(const CompositeWeightWriter &) = delete;
  CompositeWeightWriter &operator=(const CompositeWeightWriter &) = delete;

  // Writes open parenthesis to a stream if option selected.
  void WriteBegin();

  // Writes element to a stream.
  template <class T>
  void WriteElement(const T &comp) {
    if (i_++ > 0) ostrm_ << separator_;
    ostrm_ << comp;
  }

  // Writes close parenthesis to a stream if option selected.
  void WriteEnd();

 private:
  std::ostream &ostrm_;
  int i_ = 0;  // Element position.
};

// Helper class for reading textual composite weights. Elements are separated by
// a separator character. There must be at least one element per textual
// representation.  Parentheses characters should be set if the composite
// weights themselves contain composite weights to ensure proper parsing.
class CompositeWeightReader : public internal::CompositeWeightIO {
 public:
  // Uses configuration from flags (FLAGS_fst_weight_separator,
  // FLAGS_fst_weight_parentheses).
  explicit CompositeWeightReader(std::istream &istrm);

  // parentheses defines the opening and closing parenthesis characters.
  // Set parentheses = {0, 0} to disable reading parenthesis.
  CompositeWeightReader(std::istream &istrm, char separator,
                        std::pair<char, char> parentheses);

  CompositeWeightReader(const CompositeWeightReader &) = delete;
  CompositeWeightReader &operator=(const CompositeWeightReader &) = delete;

  // Reads open parenthesis from a stream if option selected.
  void ReadBegin();

  // Reads element from a stream. The second argument, when true, indicates that
  // this will be the last element (allowing more forgiving formatting of the
  // last element). Returns false when last element is read.
  template <class T>
  bool ReadElement(T *comp, bool last = false);

  // Finalizes reading.
  void ReadEnd();

 private:
  std::istream &istrm_;  // Input stream.
  int c_ = 0;            // Last character read, or EOF.
  int depth_ = 0;        // Weight parentheses depth.
};

template <class T>
inline bool CompositeWeightReader::ReadElement(T *comp, bool last) {
  string s;
  const bool has_parens = open_paren_ != 0;
  while ((c_ != std::istream::traits_type::eof()) && !std::isspace(c_) &&
         (c_ != separator_ || depth_ > 1 || last) &&
         (c_ != close_paren_ || depth_ != 1)) {
    s += c_;
    // If parentheses encountered before separator, they must be matched.
    if (has_parens && c_ == open_paren_) {
      ++depth_;
    } else if (has_parens && c_ == close_paren_) {
      // Failure on unmatched parentheses.
      if (depth_ == 0) {
        FSTERROR() << "CompositeWeightReader: Unmatched close paren: "
                   << "Is the fst_weight_parentheses flag set correctly?";
        istrm_.clear(std::ios::badbit);
        return false;
      }
      --depth_;
    }
    c_ = istrm_.get();
  }
  if (s.empty()) {
    FSTERROR() << "CompositeWeightReader: Empty element: "
               << "Is the fst_weight_parentheses flag set correctly?";
    istrm_.clear(std::ios::badbit);
    return false;
  }
  std::istringstream istrm(s);
  istrm >> *comp;
  // Skips separator/close parenthesis.
  if (c_ != std::istream::traits_type::eof() && !std::isspace(c_)) {
    c_ = istrm_.get();
  }
  const bool is_eof = c_ == std::istream::traits_type::eof();
  // Clears fail bit if just EOF.
  if (is_eof && !istrm_.bad()) istrm_.clear(std::ios::eofbit);
  return !is_eof && !std::isspace(c_);
}

}  // namespace fst

#endif  // FST_WEIGHT_H_
