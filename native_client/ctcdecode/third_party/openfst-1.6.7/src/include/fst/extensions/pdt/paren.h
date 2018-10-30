// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Common classes for PDT parentheses.

#ifndef FST_EXTENSIONS_PDT_PAREN_H_
#define FST_EXTENSIONS_PDT_PAREN_H_

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <fst/log.h>

#include <fst/extensions/pdt/collection.h>
#include <fst/extensions/pdt/pdt.h>
#include <fst/dfs-visit.h>
#include <fst/fst.h>


namespace fst {
namespace internal {

// ParenState: Pair of an open (close) parenthesis and its destination (source)
// state.

template <class Arc>
struct ParenState {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;

  Label paren_id;    // ID of open (close) paren.
  StateId state_id;  // Destination (source) state of open (close) paren.

  explicit ParenState(Label paren_id = kNoLabel, StateId state_id = kNoStateId)
      : paren_id(paren_id), state_id(state_id) {}

  bool operator==(const ParenState<Arc> &other) const {
    if (&other == this) return true;
    return other.paren_id == paren_id && other.state_id == state_id;
  }

  bool operator!=(const ParenState<Arc> &other) const {
    return !(other == *this);
  }

  struct Hash {
    size_t operator()(const ParenState<Arc> &pstate) const {
      static constexpr auto prime = 7853;
      return pstate.paren_id + pstate.state_id * prime;
    }
  };
};

// Creates an FST-style const iterator from an STL-style map.
template <class Map>
class MapIterator {
 public:
  using StlIterator = typename Map::const_iterator;
  using ValueType = typename Map::mapped_type;

  MapIterator(const Map &map, StlIterator it)
      : begin_(it), end_(map.end()), it_(it) {}

  bool Done() const { return it_ == end_ || it_->first != begin_->first; }

  ValueType Value() const { return it_->second; }

  void Next() { ++it_; }

  void Reset() { it_ = begin_; }

 private:
  const StlIterator begin_;
  const StlIterator end_;
  StlIterator it_;
};

// PdtParenReachable: Provides various parenthesis reachability information.

template <class Arc>
class PdtParenReachable {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;

  using State = ParenState<Arc>;
  using StateHash = typename State::Hash;

  // Maps from state ID to reachable paren IDs from (to) that state.
  using ParenMultimap = std::unordered_multimap<StateId, Label>;

  // Maps from paren ID and state ID to reachable state set ID.
  using StateSetMap = std::unordered_map<State, ssize_t, StateHash>;

  // Maps from paren ID and state ID to arcs exiting that state with that
  // Label.
  using ParenArcMultimap = std::unordered_map<State, Arc, StateHash>;

  using ParenIterator = MapIterator<ParenMultimap>;

  using ParenArcIterator = MapIterator<ParenArcMultimap>;

  using SetIterator = typename Collection<ssize_t, StateId>::SetIterator;

  // Computes close (open) parenthesis reachability information for a PDT with
  // bounded stack.
  PdtParenReachable(const Fst<Arc> &fst,
                    const std::vector<std::pair<Label, Label>> &parens,
                    bool close)
      : fst_(fst), parens_(parens), close_(close), error_(false) {
    paren_map_.reserve(2 * parens.size());
    for (size_t i = 0; i < parens.size(); ++i) {
      const auto &pair = parens[i];
      paren_map_[pair.first] = i;
      paren_map_[pair.second] = i;
    }
    if (close_) {
      const auto start = fst.Start();
      if (start == kNoStateId) return;
      if (!DFSearch(start)) {
        FSTERROR() << "PdtReachable: Underlying cyclicity not supported";
        error_ = true;
      }
    } else {
      FSTERROR() << "PdtParenReachable: Open paren info not implemented";
      error_ = true;
    }
  }

  bool Error() const { return error_; }

  // Given a state ID, returns an iterator over paren IDs for close (open)
  // parens reachable from that state along balanced paths.
  ParenIterator FindParens(StateId s) const {
    return ParenIterator(paren_multimap_, paren_multimap_.find(s));
  }

  // Given a paren ID and a state ID s, returns an iterator over states that can
  // be reached along balanced paths from (to) s that have have close (open)
  // parentheses matching the paren ID exiting (entering) those states.
  SetIterator FindStates(Label paren_id, StateId s) const {
    const State paren_state(paren_id, s);
    const auto it = set_map_.find(paren_state);
    if (it == set_map_.end()) {
      return state_sets_.FindSet(-1);
    } else {
      return state_sets_.FindSet(it->second);
    }
  }

  // Given a paren ID and a state ID s, return an iterator over arcs that exit
  // (enter) s and are labeled with a close (open) parenthesis matching the
  // paren ID.
  ParenArcIterator FindParenArcs(Label paren_id, StateId s) const {
    const State paren_state(paren_id, s);
    return ParenArcIterator(paren_arc_multimap_,
                            paren_arc_multimap_.find(paren_state));
  }

 private:
  // Returns false when cycle detected during DFS gathering paren and state set
  // information.
  bool DFSearch(StateId s);

  // Unions state sets together gathered by the DFS.
  void ComputeStateSet(StateId s);

  // Gathers state set(s) from state.
  void UpdateStateSet(StateId nextstate, std::set<Label> *paren_set,
                      std::vector<std::set<StateId>> *state_sets) const;

  const Fst<Arc> &fst_;
  // Paren IDs to labels.
  const std::vector<std::pair<Label, Label>> &parens_;
  // Close/open paren info?
  const bool close_;
  // Labels to paren IDs.
  std::unordered_map<Label, Label> paren_map_;
  // Paren reachability.
  ParenMultimap paren_multimap_;
  // Paren arcs.
  ParenArcMultimap paren_arc_multimap_;
  // DFS states.
  std::vector<uint8> state_color_;
  // Reachable states to IDs.
  mutable Collection<ssize_t, StateId> state_sets_;
  // IDs to reachable states.
  StateSetMap set_map_;
  bool error_;

  PdtParenReachable(const PdtParenReachable &) = delete;
  PdtParenReachable &operator=(const PdtParenReachable &) = delete;
};

// Gathers paren and state set information.
template <class Arc>
bool PdtParenReachable<Arc>::DFSearch(StateId s) {
  static constexpr uint8 kWhiteState = 0x01;  // Undiscovered.
  static constexpr uint8 kGreyState = 0x02;   // Discovered & unfinished.
  static constexpr uint8 kBlackState = 0x04;  // Finished.
  if (s >= state_color_.size()) state_color_.resize(s + 1, kWhiteState);
  if (state_color_[s] == kBlackState) return true;
  if (state_color_[s] == kGreyState) return false;
  state_color_[s] = kGreyState;
  for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
    const auto &arc = aiter.Value();
    const auto it = paren_map_.find(arc.ilabel);
    if (it != paren_map_.end()) {  // Paren?
      const auto paren_id = it->second;
      if (arc.ilabel == parens_[paren_id].first) {  // Open paren?
        if (!DFSearch(arc.nextstate)) return false;
        for (auto set_iter = FindStates(paren_id, arc.nextstate);
             !set_iter.Done(); set_iter.Next()) {
          for (auto paren_arc_iter =
                   FindParenArcs(paren_id, set_iter.Element());
               !paren_arc_iter.Done(); paren_arc_iter.Next()) {
            const auto &cparc = paren_arc_iter.Value();
            if (!DFSearch(cparc.nextstate)) return false;
          }
        }
      }
    } else if (!DFSearch(arc.nextstate)) {  // Non-paren.
      return false;
    }
  }
  ComputeStateSet(s);
  state_color_[s] = kBlackState;
  return true;
}

// Unions state sets.
template <class Arc>
void PdtParenReachable<Arc>::ComputeStateSet(StateId s) {
  std::set<Label> paren_set;
  std::vector<std::set<StateId>> state_sets(parens_.size());
  for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
    const auto &arc = aiter.Value();
    const auto it = paren_map_.find(arc.ilabel);
    if (it != paren_map_.end()) {  // Paren?
      const auto paren_id = it->second;
      if (arc.ilabel == parens_[paren_id].first) {  // Open paren?
        for (auto set_iter = FindStates(paren_id, arc.nextstate);
             !set_iter.Done(); set_iter.Next()) {
          for (auto paren_arc_iter =
                   FindParenArcs(paren_id, set_iter.Element());
               !paren_arc_iter.Done(); paren_arc_iter.Next()) {
            const auto &cparc = paren_arc_iter.Value();
            UpdateStateSet(cparc.nextstate, &paren_set, &state_sets);
          }
        }
      } else {  // Close paren.
        paren_set.insert(paren_id);
        state_sets[paren_id].insert(s);
        const State paren_state(paren_id, s);
        paren_arc_multimap_.insert(std::make_pair(paren_state, arc));
      }
    } else {  // Non-paren.
      UpdateStateSet(arc.nextstate, &paren_set, &state_sets);
    }
  }
  std::vector<StateId> state_set;
  for (auto paren_iter = paren_set.begin(); paren_iter != paren_set.end();
       ++paren_iter) {
    state_set.clear();
    const auto paren_id = *paren_iter;
    paren_multimap_.insert(std::make_pair(s, paren_id));
    for (auto state_iter = state_sets[paren_id].begin();
         state_iter != state_sets[paren_id].end(); ++state_iter) {
      state_set.push_back(*state_iter);
    }
    const State paren_state(paren_id, s);
    set_map_[paren_state] = state_sets_.FindId(state_set);
  }
}

// Gathers state sets.
template <class Arc>
void PdtParenReachable<Arc>::UpdateStateSet(
    StateId nextstate, std::set<Label> *paren_set,
    std::vector<std::set<StateId>> *state_sets) const {
  for (auto paren_iter = FindParens(nextstate); !paren_iter.Done();
       paren_iter.Next()) {
    const auto paren_id = paren_iter.Value();
    paren_set->insert(paren_id);
    for (auto set_iter = FindStates(paren_id, nextstate); !set_iter.Done();
         set_iter.Next()) {
      (*state_sets)[paren_id].insert(set_iter.Element());
    }
  }
}

// Stores balancing parenthesis data for a PDT. Unlike PdtParenReachable above
// this allows on-the-fly construction (e.g., in PdtShortestPath).
template <class Arc>
class PdtBalanceData {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;

  using State = ParenState<Arc>;
  using StateHash = typename State::Hash;

  // Set for open parens.
  using OpenParenSet = std::unordered_set<State, StateHash>;

  // Maps from open paren destination state to parenthesis ID.
  using OpenParenMap = std::unordered_multimap<StateId, Label>;

  // Maps from open paren state to source states of matching close parens
  using CloseParenMap = std::unordered_multimap<State, StateId, StateHash>;

  // Maps from open paren state to close source set ID.
  using CloseSourceMap = std::unordered_map<State, ssize_t, StateHash>;

  using SetIterator = typename Collection<ssize_t, StateId>::SetIterator;

  PdtBalanceData() {}

  void Clear() {
    open_paren_map_.clear();
    close_paren_map_.clear();
  }

  // Adds an open parenthesis with destination state open_dest.
  void OpenInsert(Label paren_id, StateId open_dest) {
    const State key(paren_id, open_dest);
    if (!open_paren_set_.count(key)) {
      open_paren_set_.insert(key);
      open_paren_map_.emplace(open_dest, paren_id);
    }
  }

  // Adds a matching closing parenthesis with source state close_source
  // balancing an open_parenthesis with destination state open_dest if
  // OpenInsert() previously called.
  void CloseInsert(Label paren_id, StateId open_dest, StateId close_source) {
    const State key(paren_id, open_dest);
    if (open_paren_set_.count(key)) {
      close_paren_map_.emplace(key, close_source);
    }
  }

  // Finds close paren source states matching an open parenthesis. The following
  // methods are then used to iterate through those matching states. Should be
  // called only after FinishInsert(open_dest).
  SetIterator Find(Label paren_id, StateId open_dest) {
    const State key(paren_id, open_dest);
    const auto it = close_source_map_.find(key);
    if (it == close_source_map_.end()) {
      return close_source_sets_.FindSet(-1);
    } else {
      return close_source_sets_.FindSet(it->second);
    }
  }

  // Called when all open and close parenthesis insertions (w.r.t. open
  // parentheses entering state open_dest) are finished. Must be called before
  // Find(open_dest).
  void FinishInsert(StateId open_dest) {
    std::vector<StateId> close_sources;
    for (auto oit = open_paren_map_.find(open_dest);
         oit != open_paren_map_.end() && oit->first == open_dest;) {
      const auto paren_id = oit->second;
      close_sources.clear();
      const State key(paren_id, open_dest);
      open_paren_set_.erase(open_paren_set_.find(key));
      for (auto cit = close_paren_map_.find(key);
           cit != close_paren_map_.end() && cit->first == key;) {
        close_sources.push_back(cit->second);
        close_paren_map_.erase(cit++);
      }
      std::sort(close_sources.begin(), close_sources.end());
      auto unique_end = std::unique(close_sources.begin(), close_sources.end());
      close_sources.resize(unique_end - close_sources.begin());
      if (!close_sources.empty()) {
        close_source_map_[key] = close_source_sets_.FindId(close_sources);
      }
      open_paren_map_.erase(oit++);
    }
  }

  // Returns a new balance data object representing the reversed balance
  // information.
  PdtBalanceData<Arc> *Reverse(StateId num_states, StateId num_split,
                               StateId state_id_shift) const;

 private:
  // Open paren at destintation state?
  OpenParenSet open_paren_set_;
  // Open parens per state.
  OpenParenMap open_paren_map_;
  // Current open destination state.
  State open_dest_;
  // Current open paren/state.
  typename OpenParenMap::const_iterator open_iter_;
  // Close states to (open paren, state).
  CloseParenMap close_paren_map_;
  // (Paren, state) to set ID.
  CloseSourceMap close_source_map_;
  mutable Collection<ssize_t, StateId> close_source_sets_;
};

// Return a new balance data object representing the reversed balance
// information.
template <class Arc>
PdtBalanceData<Arc> *PdtBalanceData<Arc>::Reverse(
    StateId num_states, StateId num_split, StateId state_id_shift) const {
  auto *bd = new PdtBalanceData<Arc>;
  std::unordered_set<StateId> close_sources;
  const auto split_size = num_states / num_split;
  for (StateId i = 0; i < num_states; i += split_size) {
    close_sources.clear();
    for (auto it = close_source_map_.begin(); it != close_source_map_.end();
         ++it) {
      const auto &okey = it->first;
      const auto open_dest = okey.state_id;
      const auto paren_id = okey.paren_id;
      for (auto set_iter = close_source_sets_.FindSet(it->second);
           !set_iter.Done(); set_iter.Next()) {
        const auto close_source = set_iter.Element();
        if ((close_source < i) || (close_source >= i + split_size)) continue;
        close_sources.insert(close_source + state_id_shift);
        bd->OpenInsert(paren_id, close_source + state_id_shift);
        bd->CloseInsert(paren_id, close_source + state_id_shift,
                        open_dest + state_id_shift);
      }
    }
    for (auto it = close_sources.begin(); it != close_sources.end(); ++it) {
      bd->FinishInsert(*it);
    }
  }
  return bd;
}

}  // namespace internal
}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_PAREN_H_
