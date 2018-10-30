// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes for representing the mapping between state tuples and state IDs.

#ifndef FST_STATE_TABLE_H_
#define FST_STATE_TABLE_H_

#include <deque>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/bi-table.h>
#include <fst/expanded-fst.h>
#include <fst/filter-state.h>


namespace fst {

// State tables determine the bijective mapping between state tuples (e.g., in
// composition, triples of two FST states and a composition filter state) and
// their corresponding state IDs. They are classes, templated on state tuples,
// with the following interface:
//
// template <class T>
// class StateTable {
//  public:
//   using StateTuple = T;
//
//   // Required constructors.
//   StateTable();
//
//   StateTable(const StateTable &);
//
//   // Looks up state ID by tuple. If it doesn't exist, then add it.
//   StateId FindState(const StateTuple &tuple);
//
//   // Looks up state tuple by state ID.
//   const StateTuple<StateId> &Tuple(StateId s) const;
//
//   // # of stored tuples.
//   StateId Size() const;
// };
//
// A state tuple has the form:
//
// template <class S>
// struct StateTuple {
//   using StateId = S;
//
//   // Required constructors.
//
//   StateTuple();
//
//   StateTuple(const StateTuple &tuple);
// };

// An implementation using a hash map for the tuple to state ID mapping. The
// state tuple T must support operator==.
template <class T, class H>
class HashStateTable : public HashBiTable<typename T::StateId, T, H> {
 public:
  using StateTuple = T;
  using StateId = typename StateTuple::StateId;

  using HashBiTable<StateId, StateTuple, H>::FindId;
  using HashBiTable<StateId, StateTuple, H>::FindEntry;
  using HashBiTable<StateId, StateTuple, H>::Size;

  HashStateTable() : HashBiTable<StateId, StateTuple, H>() {}

  explicit HashStateTable(size_t table_size)
      : HashBiTable<StateId, StateTuple, H>(table_size) {}

  StateId FindState(const StateTuple &tuple) { return FindId(tuple); }

  const StateTuple &Tuple(StateId s) const { return FindEntry(s); }
};

// An implementation using a hash map for the tuple to state ID mapping. The
// state tuple T must support operator==.
template <class T, class H>
class CompactHashStateTable
    : public CompactHashBiTable<typename T::StateId, T, H> {
 public:
  using StateTuple = T;
  using StateId = typename StateTuple::StateId;

  using CompactHashBiTable<StateId, StateTuple, H>::FindId;
  using CompactHashBiTable<StateId, StateTuple, H>::FindEntry;
  using CompactHashBiTable<StateId, StateTuple, H>::Size;

  CompactHashStateTable() : CompactHashBiTable<StateId, StateTuple, H>() {}

  explicit CompactHashStateTable(size_t table_size)
      : CompactHashBiTable<StateId, StateTuple, H>(table_size) {}

  StateId FindState(const StateTuple &tuple) { return FindId(tuple); }

  const StateTuple &Tuple(StateId s) const { return FindEntry(s); }
};

// An implementation using a vector for the tuple to state mapping. It is
// passed a fingerprint functor that should fingerprint tuples uniquely to an
// integer that can used as a vector index. Normally, VectorStateTable
// constructs the fingerprint functor. Alternately, the user can pass this
// object, in which case the table takes ownership.
template <class T, class FP>
class VectorStateTable : public VectorBiTable<typename T::StateId, T, FP> {
 public:
  using StateTuple = T;
  using StateId = typename StateTuple::StateId;

  using VectorBiTable<StateId, StateTuple, FP>::FindId;
  using VectorBiTable<StateId, StateTuple, FP>::FindEntry;
  using VectorBiTable<StateId, StateTuple, FP>::Size;
  using VectorBiTable<StateId, StateTuple, FP>::Fingerprint;

  explicit VectorStateTable(FP *fingerprint = nullptr, size_t table_size = 0)
      : VectorBiTable<StateId, StateTuple, FP>(fingerprint, table_size) {}

  StateId FindState(const StateTuple &tuple) { return FindId(tuple); }

  const StateTuple &Tuple(StateId s) const { return FindEntry(s); }
};

// An implementation using a vector and a compact hash table. The selection
// functor returns true for tuples to be hashed in the vector. The fingerprint
// functor should fingerprint tuples uniquely to an integer that can be used as
// a vector index. A hash functor is used when hashing tuples into the compact
// hash table.
template <class T, class Select, class FP, class H>
class VectorHashStateTable
    : public VectorHashBiTable<typename T::StateId, T, Select, FP, H> {
 public:
  using StateTuple = T;
  using StateId = typename StateTuple::StateId;

  using VectorHashBiTable<StateId, StateTuple, Select, FP, H>::FindId;
  using VectorHashBiTable<StateId, StateTuple, Select, FP, H>::FindEntry;
  using VectorHashBiTable<StateId, StateTuple, Select, FP, H>::Size;
  using VectorHashBiTable<StateId, StateTuple, Select, FP, H>::Selector;
  using VectorHashBiTable<StateId, StateTuple, Select, FP, H>::Fingerprint;
  using VectorHashBiTable<StateId, StateTuple, Select, FP, H>::Hash;

  VectorHashStateTable(Select *select, FP *fingerprint, H *hash,
                       size_t vector_size = 0, size_t tuple_size = 0)
      : VectorHashBiTable<StateId, StateTuple, Select, FP, H>(
            select, fingerprint, hash, vector_size, tuple_size) {}

  StateId FindState(const StateTuple &tuple) { return FindId(tuple); }

  const StateTuple &Tuple(StateId s) const { return FindEntry(s); }
};

// An implementation using a hash map to map from tuples to state IDs. This
// version permits erasing of states. The state tuple's default constructor
// must produce a tuple that will never be seen and the table must suppor
// operator==.
template <class T, class H>
class ErasableStateTable : public ErasableBiTable<typename T::StateId, T, H> {
 public:
  using StateTuple = T;
  using StateId = typename StateTuple::StateId;

  using ErasableBiTable<StateId, StateTuple, H>::FindId;
  using ErasableBiTable<StateId, StateTuple, H>::FindEntry;
  using ErasableBiTable<StateId, StateTuple, H>::Size;
  using ErasableBiTable<StateId, StateTuple, H>::Erase;

  ErasableStateTable() : ErasableBiTable<StateId, StateTuple, H>() {}

  StateId FindState(const StateTuple &tuple) { return FindId(tuple); }

  const StateTuple &Tuple(StateId s) const { return FindEntry(s); }
};

// The composition state table has the form:
//
// template <class Arc, class FilterState>
// class ComposeStateTable {
//  public:
//   using StateId = typename Arc::StateId;
//
//   // Required constructors.
//
//   ComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2);
//   ComposeStateTable(const ComposeStateTable<Arc, FilterState> &table);
//
//   // Looks up a state ID by tuple, adding it if doesn't exist.
//   StateId FindState(const StateTuple &tuple);
//
//   // Looks up a tuple by state ID.
//   const ComposeStateTuple<StateId> &Tuple(StateId s) const;
//
//   // The number of of stored tuples.
//   StateId Size() const;
//
//   // Return true if error was encountered.
//   bool Error() const;
// };
//
// The following interface is used to represent the composition state.
//
// template <class S, class FS>
// class CompositionStateTuple {
//  public:
//   using StateId = typename StateId;
//   using FS = FilterState;
//
//   // Required constructors.
//   StateTuple();
//   StateTuple(StateId s1, StateId s2, const FilterState &fs);
//
//   StateId StateId1() const;
//   StateId StateId2() const;
//
//   FilterState GetFilterState() const;
//
//   std::pair<StateId, StateId> StatePair() const;
//
//   size_t Hash() const;
//
//   friend bool operator==(const StateTuple& x, const StateTuple &y);
// }
//
template <typename S, typename FS>
class DefaultComposeStateTuple {
 public:
  using StateId = S;
  using FilterState = FS;

  DefaultComposeStateTuple()
      : state_pair_(kNoStateId, kNoStateId), fs_(FilterState::NoState()) {}

  DefaultComposeStateTuple(StateId s1, StateId s2, const FilterState &fs)
      : state_pair_(s1, s2), fs_(fs) {}

  StateId StateId1() const { return state_pair_.first; }

  StateId StateId2() const { return state_pair_.second; }

  FilterState GetFilterState() const { return fs_; }

  const std::pair<StateId, StateId> &StatePair() const { return state_pair_; }

  friend bool operator==(const DefaultComposeStateTuple &x,
                         const DefaultComposeStateTuple &y) {
    return (&x == &y) || (x.state_pair_ == y.state_pair_ && x.fs_ == y.fs_);
  }

  size_t Hash() const {
    return static_cast<size_t>(StateId1()) +
           static_cast<size_t>(StateId2()) * 7853u +
           GetFilterState().Hash() * 7867u;
  }

 private:
  std::pair<StateId, StateId> state_pair_;
  FilterState fs_;  // State of composition filter.
};

// Specialization for TrivialFilterState that does not explicitely store the
// filter state since it is always the unique non-blocking state.
template <typename S>
class DefaultComposeStateTuple<S, TrivialFilterState> {
 public:
  using StateId = S;
  using FilterState = TrivialFilterState;

  DefaultComposeStateTuple()
      : state_pair_(kNoStateId, kNoStateId) {}

  DefaultComposeStateTuple(StateId s1, StateId s2, const FilterState &)
      : state_pair_(s1, s2) {}

  StateId StateId1() const { return state_pair_.first; }

  StateId StateId2() const { return state_pair_.second; }

  FilterState GetFilterState() const { return FilterState(true); }

  const std::pair<StateId, StateId> &StatePair() const { return state_pair_; }

  friend bool operator==(const DefaultComposeStateTuple &x,
                         const DefaultComposeStateTuple &y) {
    return (&x == &y) || (x.state_pair_ == y.state_pair_);
  }

  size_t Hash() const { return StateId1() + StateId2() * 7853; }

 private:
  std::pair<StateId, StateId> state_pair_;
};

// Hashing of composition state tuples.
template <typename T>
class ComposeHash {
 public:
  size_t operator()(const T &t) const { return t.Hash(); }
};

// A HashStateTable over composition tuples.
template <typename Arc, typename FilterState,
          typename StateTuple =
              DefaultComposeStateTuple<typename Arc::StateId, FilterState>,
          typename StateTable =
              CompactHashStateTable<StateTuple, ComposeHash<StateTuple>>>
class GenericComposeStateTable : public StateTable {
 public:
  using StateId = typename Arc::StateId;

  GenericComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {}

  GenericComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                           size_t table_size)
      : StateTable(table_size) {}

  constexpr bool Error() const { return false; }

 private:
  GenericComposeStateTable &operator=(const GenericComposeStateTable &table) =
      delete;
};

//  Fingerprint for general composition tuples.
template <typename StateTuple>
class ComposeFingerprint {
 public:
  using StateId = typename StateTuple::StateId;

  // Required but suboptimal constructor.
  ComposeFingerprint() : mult1_(8192), mult2_(8192) {
    LOG(WARNING) << "TupleFingerprint: # of FST states should be provided.";
  }

  // Constructor is provided the sizes of the input FSTs.
  ComposeFingerprint(StateId nstates1, StateId nstates2)
      : mult1_(nstates1), mult2_(nstates1 * nstates2) {}

  size_t operator()(const StateTuple &tuple) {
    return tuple.StateId1() + tuple.StateId2() * mult1_ +
           tuple.GetFilterState().Hash() * mult2_;
  }

 private:
  const ssize_t mult1_;
  const ssize_t mult2_;
};

// Useful when the first composition state determines the tuple.
template <typename StateTuple>
class ComposeState1Fingerprint {
 public:
  size_t operator()(const StateTuple &tuple) { return tuple.StateId1(); }
};

// Useful when the second composition state determines the tuple.
template <typename StateTuple>
class ComposeState2Fingerprint {
 public:
  size_t operator()(const StateTuple &tuple) { return tuple.StateId2(); }
};

// A VectorStateTable over composition tuples. This can be used when the
// product of number of states in FST1 and FST2 (and the composition filter
// state hash) is manageable. If the FSTs are not expanded FSTs, they will
// first have their states counted.
template <typename Arc, typename StateTuple>
class ProductComposeStateTable
    : public VectorStateTable<StateTuple, ComposeFingerprint<StateTuple>> {
 public:
  using StateId = typename Arc::StateId;
  using StateTable =
      VectorStateTable<StateTuple, ComposeFingerprint<StateTuple>>;

  ProductComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                           size_t table_size = 0)
      : StateTable(new ComposeFingerprint<StateTuple>(CountStates(fst1),
                                                      CountStates(fst2)),
                   table_size) {}

  ProductComposeStateTable(
      const ProductComposeStateTable<Arc, StateTuple> &table)
      : StateTable(new ComposeFingerprint<StateTuple>(table.Fingerprint())) {}

  constexpr bool Error() const { return false; }

 private:
  ProductComposeStateTable &operator=(const ProductComposeStateTable &table) =
      delete;
};

// A vector-backed table over composition tuples which can be used when the
// first FST is a string (i.e., satisfies kString property) and the second is
// deterministic and epsilon-free. It should be used with a composition filter
// that creates at most one filter state per tuple under these conditions (e.g.,
// SequenceComposeFilter or MatchComposeFilter).
template <typename Arc, typename StateTuple>
class StringDetComposeStateTable
    : public VectorStateTable<StateTuple,
                              ComposeState1Fingerprint<StateTuple>> {
 public:
  using StateId = typename Arc::StateId;
  using StateTable =
      VectorStateTable<StateTuple, ComposeState1Fingerprint<StateTuple>>;

  StringDetComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2)
      : error_(false) {
    static constexpr auto props2 = kIDeterministic | kNoIEpsilons;
    if (fst1.Properties(kString, true) != kString) {
      FSTERROR() << "StringDetComposeStateTable: 1st FST is not a string";
      error_ = true;
    } else if (fst2.Properties(props2, true) != props2) {
      FSTERROR() << "StringDetComposeStateTable: 2nd FST is not deterministic "
                    "and epsilon-free";
      error_ = true;
    }
  }

  StringDetComposeStateTable(
      const StringDetComposeStateTable<Arc, StateTuple> &table)
      : StateTable(table), error_(table.error_) {}

  bool Error() const { return error_; }

 private:
  bool error_;

  StringDetComposeStateTable &operator=(const StringDetComposeStateTable &) =
      delete;
};

// A vector-backed table over composition tuples which can be used when the
// first FST is deterministic and epsilon-free and the second is a string (i.e.,
// satisfies kString). It should be used with a composition filter that creates
// at most one filter state per tuple under these conditions (e.g.,
// SequenceComposeFilter or MatchComposeFilter).
template <typename Arc, typename StateTuple>
class DetStringComposeStateTable
    : public VectorStateTable<StateTuple,
                              ComposeState2Fingerprint<StateTuple>> {
 public:
  using StateId = typename Arc::StateId;
  using StateTable =
      VectorStateTable<StateTuple, ComposeState2Fingerprint<StateTuple>>;

  DetStringComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2)
      : error_(false) {
    static constexpr auto props = kODeterministic | kNoOEpsilons;
    if (fst1.Properties(props, true) != props) {
      FSTERROR() << "StringDetComposeStateTable: 1st FST is not "
                 << "input-deterministic and epsilon-free";
      error_ = true;
    } else if (fst2.Properties(kString, true) != kString) {
      FSTERROR() << "DetStringComposeStateTable: 2nd FST is not a string";
      error_ = true;
    }
  }

  DetStringComposeStateTable(
      const DetStringComposeStateTable<Arc, StateTuple> &table)
      : StateTable(table), error_(table.error_) {}

  bool Error() const { return error_; }

 private:
  bool error_;

  DetStringComposeStateTable &operator=(const DetStringComposeStateTable &) =
      delete;
};

// An erasable table over composition tuples. The Erase(StateId) method can be
// called if the user either is sure that composition will never return to that
// tuple or doesn't care that if it does, it is assigned a new state ID.
template <typename Arc, typename StateTuple>
class ErasableComposeStateTable
    : public ErasableStateTable<StateTuple, ComposeHash<StateTuple>> {
 public:
  ErasableComposeStateTable(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {}

  constexpr bool Error() const { return false; }

 private:
  ErasableComposeStateTable &operator=(const ErasableComposeStateTable &table) =
      delete;
};

}  // namespace fst

#endif  // FST_STATE_TABLE_H_
