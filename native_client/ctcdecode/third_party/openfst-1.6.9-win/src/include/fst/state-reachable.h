// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to determine whether a given (final) state can be reached from some
// other given state.

#ifndef FST_STATE_REACHABLE_H_
#define FST_STATE_REACHABLE_H_

#include <vector>

#include <fst/log.h>

#include <fst/connect.h>
#include <fst/dfs-visit.h>
#include <fst/fst.h>
#include <fst/interval-set.h>
#include <fst/vector-fst.h>


namespace fst {

// Computes the (final) states reachable from a given state in an FST. After
// this visitor has been called, a final state f can be reached from a state
// s iff (*isets)[s].Member(state2index[f]) is true, where (*isets[s]) is a
// set of half-open inteval of final state indices and state2index[f] maps from
// a final state to its index. If state2index is empty, it is filled-in with
// suitable indices. If it is non-empty, those indices are used; in this case,
// the final states must have out-degree 0.
template <class Arc, class I = typename Arc::StateId, class S = IntervalSet<I>>
class IntervalReachVisitor {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Index = I;
  using ISet = S;
  using Interval = typename ISet::Interval;

  IntervalReachVisitor(const Fst<Arc> &fst, std::vector<S> *isets,
                       std::vector<Index> *state2index)
      : fst_(fst),
        isets_(isets),
        state2index_(state2index),
        index_(state2index->empty() ? 1 : -1),
        error_(false) {
    isets_->clear();
  }

  void InitVisit(const Fst<Arc> &) { error_ = false; }

  bool InitState(StateId s, StateId r) {
    while (isets_->size() <= s) isets_->push_back(S());
    while (state2index_->size() <= s) state2index_->push_back(-1);
    if (fst_.Final(s) != Weight::Zero()) {
      // Create tree interval.
      auto *intervals = (*isets_)[s].MutableIntervals();
      if (index_ < 0) {  // Uses state2index_ map to set index.
        if (fst_.NumArcs(s) > 0) {
          FSTERROR() << "IntervalReachVisitor: state2index map must be empty "
                     << "for this FST";
          error_ = true;
          return false;
        }
        const auto index = (*state2index_)[s];
        if (index < 0) {
          FSTERROR() << "IntervalReachVisitor: state2index map incomplete";
          error_ = true;
          return false;
        }
        intervals->push_back(Interval(index, index + 1));
      } else {  // Use pre-order index.
        intervals->push_back(Interval(index_, index_ + 1));
        (*state2index_)[s] = index_++;
      }
    }
    return true;
  }

  constexpr bool TreeArc(StateId, const Arc &) const { return true; }

  bool BackArc(StateId s, const Arc &arc) {
    FSTERROR() << "IntervalReachVisitor: Cyclic input";
    error_ = true;
    return false;
  }

  bool ForwardOrCrossArc(StateId s, const Arc &arc) {
    // Non-tree interval.
    (*isets_)[s].Union((*isets_)[arc.nextstate]);
    return true;
  }

  void FinishState(StateId s, StateId p, const Arc *) {
    if (index_ >= 0 && fst_.Final(s) != Weight::Zero()) {
      auto *intervals = (*isets_)[s].MutableIntervals();
      (*intervals)[0].end = index_;  // Updates tree interval end.
    }
    (*isets_)[s].Normalize();
    if (p != kNoStateId) {
      (*isets_)[p].Union((*isets_)[s]);  // Propagates intervals to parent.
    }
  }

  void FinishVisit() {}

  bool Error() const { return error_; }

 private:
  const Fst<Arc> &fst_;
  std::vector<ISet> *isets_;
  std::vector<Index> *state2index_;
  Index index_;
  bool error_;
};

// Tests reachability of final states from a given state. To test for
// reachability from a state s, first do SetState(s). Then a final state f can
// be reached from state s of FST iff Reach(f) is true. The input can be cyclic,
// but no cycle may contain a final state.
template <class Arc, class I = typename Arc::StateId, class S = IntervalSet<I>>
class StateReachable {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Index = I;
  using ISet = S;
  using Interval = typename ISet::Interval;

  explicit StateReachable(const Fst<Arc> &fst) : error_(false) {
    if (fst.Properties(kAcyclic, true)) {
      AcyclicStateReachable(fst);
    } else {
      CyclicStateReachable(fst);
    }
  }

  explicit StateReachable(const StateReachable<Arc> &reachable) {
    FSTERROR() << "Copy constructor for state reachable class "
               << "not implemented.";
    error_ = true;
  }

  // Sets current state.
  void SetState(StateId s) { s_ = s; }

  // Can reach this final state from current state?
  bool Reach(StateId s) {
    if (s >= state2index_.size()) return false;
    const auto i = state2index_[s];
    if (i < 0) {
      FSTERROR() << "StateReachable: State non-final: " << s;
      error_ = true;
      return false;
    }
    return isets_[s_].Member(i);
  }

  // Access to the state-to-index mapping. Unassigned states have index -1.
  std::vector<Index> &State2Index() { return state2index_; }

  // Access to the interval sets. These specify the reachability to the final
  // states as intervals of the final state indices.
  const std::vector<ISet> &IntervalSets() { return isets_; }

  bool Error() const { return error_; }

 private:
  void AcyclicStateReachable(const Fst<Arc> &fst) {
    IntervalReachVisitor<Arc, StateId, ISet> reach_visitor(fst, &isets_,
                                                           &state2index_);
    DfsVisit(fst, &reach_visitor);
    if (reach_visitor.Error()) error_ = true;
  }

  void CyclicStateReachable(const Fst<Arc> &fst) {
    // Finds state reachability on the acyclic condensation FST.
    VectorFst<Arc> cfst;
    std::vector<StateId> scc;
    Condense(fst, &cfst, &scc);
    StateReachable reachable(cfst);
    if (reachable.Error()) {
      error_ = true;
      return;
    }
    // Gets the number of states per SCC.
    std::vector<size_t> nscc;
    for (StateId s = 0; s < scc.size(); ++s) {
      const auto c = scc[s];
      while (c >= nscc.size()) nscc.push_back(0);
      ++nscc[c];
    }
    // Constructs the interval sets and state index mapping for the original
    // FST from the condensation FST.
    state2index_.resize(scc.size(), -1);
    isets_.resize(scc.size());
    for (StateId s = 0; s < scc.size(); ++s) {
      const auto c = scc[s];
      isets_[s] = reachable.IntervalSets()[c];
      state2index_[s] = reachable.State2Index()[c];
      // Checks that each final state in an input FST is not contained in a
      // cycle (i.e., not in a non-trivial SCC).
      if (cfst.Final(c) != Weight::Zero() && nscc[c] > 1) {
        FSTERROR() << "StateReachable: Final state contained in a cycle";
        error_ = true;
        return;
      }
    }
  }

  StateId s_;                       // Current state.
  std::vector<ISet> isets_;         // Interval sets per state.
  std::vector<Index> state2index_;  // Finds index for a final state.
  bool error_;

  StateReachable &operator=(const StateReachable &) = delete;
};

}  // namespace fst

#endif  // FST_STATE_REACHABLE_H_
