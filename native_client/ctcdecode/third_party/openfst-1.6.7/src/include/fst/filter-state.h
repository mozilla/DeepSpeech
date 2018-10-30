// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes for storing filter state in various algorithms like composition.

#ifndef FST_FILTER_STATE_H_
#define FST_FILTER_STATE_H_

#include <forward_list>
#include <utility>

#include <fst/fst-decl.h>  // For optional argument declarations
#include <fst/fst.h>
#include <fst/matcher.h>


namespace fst {

// The filter state interface represents the state of a (e.g., composition)
// filter.
//
// class FilterState {
//  public:
//   // Required constructors.
//
//   FilterState();
//
//   FilterState(const FilterState &fs);
//
//   // An invalid filter state.
//   static const FilterState NoState();
//
//   // Maps state to integer for hashing.
//   size_t Hash() const;
//
//   // Equality of filter states.
//   bool operator==(const FilterState &fs) const;
//
//   // Inequality of filter states.
//   bool operator!=(const FilterState &fs) const;
//
//   // Assignment to filter states.
//   FilterState &operator=(const FilterState& fs);
// };

// Filter state that is a signed integral type.
template <typename T>
class IntegerFilterState {
 public:
  IntegerFilterState() : state_(kNoStateId) {}

  explicit IntegerFilterState(T s) : state_(s) {}

  static const IntegerFilterState NoState() { return IntegerFilterState(); }

  size_t Hash() const { return static_cast<size_t>(state_); }

  bool operator==(const IntegerFilterState &fs) const {
    return state_ == fs.state_;
  }

  bool operator!=(const IntegerFilterState &fs) const {
    return state_ != fs.state_;
  }

  T GetState() const { return state_; }

  void SetState(T state) { state_ = state; }

 private:
  T state_;
};

using CharFilterState = IntegerFilterState<signed char>;
using ShortFilterState = IntegerFilterState<short>;  // NOLINT
using IntFilterState = IntegerFilterState<int>;

// Filter state that is a weight (class).
template <class W>
class WeightFilterState {
 public:
  WeightFilterState() : weight_(W::Zero()) {}

  explicit WeightFilterState(W weight) : weight_(std::move(weight)) {}

  static const WeightFilterState NoState() { return WeightFilterState(); }

  size_t Hash() const { return weight_.Hash(); }

  bool operator==(const WeightFilterState &fs) const {
    return weight_ == fs.weight_;
  }

  bool operator!=(const WeightFilterState &fs) const {
    return weight_ != fs.weight_;
  }

  W GetWeight() const { return weight_; }

  void SetWeight(W weight) { weight_ = std::move(weight); }

 private:
  W weight_;
};

// Filter state is a list of signed integer types T. Order matters
// for equality.
template <typename T>
class ListFilterState {
 public:
  ListFilterState() {}

  explicit ListFilterState(T s) { list_.push_front(s); }

  static const ListFilterState NoState() { return ListFilterState(kNoStateId); }

  size_t Hash() const {
    size_t h = 0;
    for (const auto &elem : list_) h ^= h << 1 ^ elem;
    return h;
  }

  bool operator==(const ListFilterState &fs) const { return list_ == fs.list_; }

  bool operator!=(const ListFilterState &fs) const { return list_ != fs.list_; }

  const std::forward_list<T> &GetState() const { return list_; }

  std::forward_list<T> *GetMutableState() { return &list_; }

  void SetState(const std::forward_list<T> &state) { list_ = state; }

 private:
  std::forward_list<T> list_;
};

// Filter state that is the combination of two filter states.
template <class FS1, class FS2>
class PairFilterState {
 public:
  PairFilterState() : fs1_(FS1::NoState()), fs2_(FS2::NoState()) {}

  PairFilterState(const FS1 &fs1, const FS2 &fs2) : fs1_(fs1), fs2_(fs2) {}

  static const PairFilterState NoState() { return PairFilterState(); }

  size_t Hash() const {
    const auto h1 = fs1_.Hash();
    static constexpr auto lshift = 5;
    static constexpr auto rshift = CHAR_BIT * sizeof(size_t) - 5;
    return h1 << lshift ^ h1 >> rshift ^ fs2_.Hash();
  }

  bool operator==(const PairFilterState &fs) const {
    return fs1_ == fs.fs1_ && fs2_ == fs.fs2_;
  }

  bool operator!=(const PairFilterState &fs) const {
    return fs1_ != fs.fs1_ || fs2_ != fs.fs2_;
  }

  const FS1 &GetState1() const { return fs1_; }

  const FS2 &GetState2() const { return fs2_; }

  void SetState(const FS1 &fs1, const FS2 &fs2) {
    fs1_ = fs1;
    fs2_ = fs2;
  }

 private:
  FS1 fs1_;
  FS2 fs2_;
};

// Single non-blocking filter state.
class TrivialFilterState {
 public:
  explicit TrivialFilterState(bool state = false) : state_(state) {}

  static const TrivialFilterState NoState() { return TrivialFilterState(); }

  size_t Hash() const { return 0; }

  bool operator==(const TrivialFilterState &fs) const {
    return state_ == fs.state_;
  }

  bool operator!=(const TrivialFilterState &fs) const {
    return state_ != fs.state_;
  }

 private:
  bool state_;
};

}  // namespace fst

#endif  // FST_FILTER_STATE_H_
