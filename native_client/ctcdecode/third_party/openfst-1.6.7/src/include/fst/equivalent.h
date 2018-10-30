// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to determine the equivalence of two FSTs.

#ifndef FST_EQUIVALENT_H_
#define FST_EQUIVALENT_H_

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fst/log.h>

#include <fst/encode.h>
#include <fst/push.h>
#include <fst/union-find.h>
#include <fst/vector-fst.h>


namespace fst {
namespace internal {

// Traits-like struct holding utility functions/typedefs/constants for
// the equivalence algorithm.
//
// Encoding device: in order to make the statesets of the two acceptors
// disjoint, we map Arc::StateId on the type MappedId. The states of
// the first acceptor are mapped on odd numbers (s -> 2s + 1), and
// those of the second one on even numbers (s -> 2s + 2). The number 0
// is reserved for an implicit (non-final) dead state (required for
// the correct treatment of non-coaccessible states; kNoStateId is mapped to
// kDeadState for both acceptors). The union-find algorithm operates on the
// mapped IDs.
template <class Arc>
struct EquivalenceUtil {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using MappedId = StateId;  // ID for an equivalence class.

  // MappedId for an implicit dead state.
  static constexpr MappedId kDeadState = 0;

  // MappedId for lookup failure.
  static constexpr MappedId kInvalidId = -1;

  // Maps state ID to the representative of the corresponding
  // equivalence class. The parameter 'which_fst' takes the values 1
  // and 2, identifying the input FST.
  static MappedId MapState(StateId s, int32 which_fst) {
    return (kNoStateId == s) ? kDeadState
                             : (static_cast<MappedId>(s) << 1) + which_fst;
  }

  // Maps set ID to State ID.
  static StateId UnMapState(MappedId id) {
    return static_cast<StateId>((--id) >> 1);
  }

  // Convenience function: checks if state with MappedId s is final in
  // acceptor fa.
  static bool IsFinal(const Fst<Arc> &fa, MappedId s) {
    return (kDeadState == s) ? false
                             : (fa.Final(UnMapState(s)) != Weight::Zero());
  }
  // Convenience function: returns the representative of ID in sets,
  // creating a new set if needed.
  static MappedId FindSet(UnionFind<MappedId> *sets, MappedId id) {
    const auto repr = sets->FindSet(id);
    if (repr != kInvalidId) {
      return repr;
    } else {
      sets->MakeSet(id);
      return id;
    }
  }
};

template <class Arc>
constexpr
    typename EquivalenceUtil<Arc>::MappedId EquivalenceUtil<Arc>::kDeadState;

template <class Arc>
constexpr
    typename EquivalenceUtil<Arc>::MappedId EquivalenceUtil<Arc>::kInvalidId;

}  // namespace internal

// Equivalence checking algorithm: determines if the two FSTs fst1 and fst2
// are equivalent. The input FSTs must be deterministic input-side epsilon-free
// acceptors, unweighted or with weights over a left semiring. Two acceptors are
// considered equivalent if they accept exactly the same set of strings (with
// the same weights).
//
// The algorithm (cf. Aho, Hopcroft and Ullman, "The Design and Analysis of
// Computer Programs") successively constructs sets of states that can be
// reached by the same prefixes, starting with a set containing the start states
// of both acceptors. A disjoint tree forest (the union-find algorithm) is used
// to represent the sets of states. The algorithm returns false if one of the
// constructed sets contains both final and non-final states. Returns an
// optional error value (useful when FLAGS_error_fatal = false).
//
// Complexity:
//
// Quasi-linear, i.e., O(n G(n)), where
//
//   n = |S1| + |S2| is the number of states in both acceptors
//
//   G(n) is a very slowly growing function that can be approximated
//        by 4 by all practical purposes.
template <class Arc>
bool Equivalent(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                float delta = kDelta, bool *error = nullptr) {
  using Weight = typename Arc::Weight;
  if (error) *error = false;
  // Check that the symbol table are compatible.
  if (!CompatSymbols(fst1.InputSymbols(), fst2.InputSymbols()) ||
      !CompatSymbols(fst1.OutputSymbols(), fst2.OutputSymbols())) {
    FSTERROR() << "Equivalent: Input/output symbol tables of 1st argument "
               << "do not match input/output symbol tables of 2nd argument";
    if (error) *error = true;
    return false;
  }
  // Check properties first.
  static constexpr auto props = kNoEpsilons | kIDeterministic | kAcceptor;
  if (fst1.Properties(props, true) != props) {
    FSTERROR() << "Equivalent: 1st argument not an"
               << " epsilon-free deterministic acceptor";
    if (error) *error = true;
    return false;
  }
  if (fst2.Properties(props, true) != props) {
    FSTERROR() << "Equivalent: 2nd argument not an"
               << " epsilon-free deterministic acceptor";
    if (error) *error = true;
    return false;
  }
  if ((fst1.Properties(kUnweighted, true) != kUnweighted) ||
      (fst2.Properties(kUnweighted, true) != kUnweighted)) {
    VectorFst<Arc> efst1(fst1);
    VectorFst<Arc> efst2(fst2);
    Push(&efst1, REWEIGHT_TO_INITIAL, delta);
    Push(&efst2, REWEIGHT_TO_INITIAL, delta);
    ArcMap(&efst1, QuantizeMapper<Arc>(delta));
    ArcMap(&efst2, QuantizeMapper<Arc>(delta));
    EncodeMapper<Arc> mapper(kEncodeWeights | kEncodeLabels, ENCODE);
    ArcMap(&efst1, &mapper);
    ArcMap(&efst2, &mapper);
    return Equivalent(efst1, efst2);
  }
  using Util = internal::EquivalenceUtil<Arc>;
  using MappedId = typename Util::MappedId;
  enum { FST1 = 1, FST2 = 2 };  // Required by Util::MapState(...)
  auto s1 = Util::MapState(fst1.Start(), FST1);
  auto s2 = Util::MapState(fst2.Start(), FST2);
  // The union-find structure.
  UnionFind<MappedId> eq_classes(1000, Util::kInvalidId);
  // Initializes the union-find structure.
  eq_classes.MakeSet(s1);
  eq_classes.MakeSet(s2);
  // Data structure for the (partial) acceptor transition function of fst1 and
  // fst2: input labels mapped to pairs of MappedIds representing destination
  // states of the corresponding arcs in fst1 and fst2, respectively.
  using Label2StatePairMap =
      std::unordered_map<typename Arc::Label, std::pair<MappedId, MappedId>>;
  Label2StatePairMap arc_pairs;
  // Pairs of MappedId's to be processed, organized in a queue.
  std::deque<std::pair<MappedId, MappedId>> q;
  bool ret = true;
  // Returns early if the start states differ w.r.t. finality.
  if (Util::IsFinal(fst1, s1) != Util::IsFinal(fst2, s2)) ret = false;
  // Main loop: explores the two acceptors in a breadth-first manner, updating
  // the equivalence relation on the statesets. Loop invariant: each block of
  // the states contains either final states only or non-final states only.
  for (q.push_back(std::make_pair(s1, s2)); ret && !q.empty(); q.pop_front()) {
    s1 = q.front().first;
    s2 = q.front().second;
    // Representatives of the equivalence classes of s1/s2.
    const auto rep1 = Util::FindSet(&eq_classes, s1);
    const auto rep2 = Util::FindSet(&eq_classes, s2);
    if (rep1 != rep2) {
      eq_classes.Union(rep1, rep2);
      arc_pairs.clear();
      // Copies outgoing arcs starting at s1 into the hash-table.
      if (Util::kDeadState != s1) {
        ArcIterator<Fst<Arc>> arc_iter(fst1, Util::UnMapState(s1));
        for (; !arc_iter.Done(); arc_iter.Next()) {
          const auto &arc = arc_iter.Value();
          // Zero-weight arcs are treated as if they did not exist.
          if (arc.weight != Weight::Zero()) {
            arc_pairs[arc.ilabel].first = Util::MapState(arc.nextstate, FST1);
          }
        }
      }
      // Copies outgoing arcs starting at s2 into the hashtable.
      if (Util::kDeadState != s2) {
        ArcIterator<Fst<Arc>> arc_iter(fst2, Util::UnMapState(s2));
        for (; !arc_iter.Done(); arc_iter.Next()) {
          const auto &arc = arc_iter.Value();
          // Zero-weight arcs are treated as if they did not exist.
          if (arc.weight != Weight::Zero()) {
            arc_pairs[arc.ilabel].second = Util::MapState(arc.nextstate, FST2);
          }
        }
      }
      // Iterates through the hashtable and process pairs of target states.
      for (const auto &arc_iter : arc_pairs) {
        const auto &pair = arc_iter.second;
        if (Util::IsFinal(fst1, pair.first) !=
            Util::IsFinal(fst2, pair.second)) {
          // Detected inconsistency: return false.
          ret = false;
          break;
        }
        q.push_back(pair);
      }
    }
  }
  if (fst1.Properties(kError, false) || fst2.Properties(kError, false)) {
    if (error) *error = true;
    return false;
  }
  return ret;
}

}  // namespace fst

#endif  // FST_EQUIVALENT_H_
