// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions to find shortest paths in a PDT.

#ifndef FST_EXTENSIONS_PDT_SHORTEST_PATH_H_
#define FST_EXTENSIONS_PDT_SHORTEST_PATH_H_

#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/extensions/pdt/paren.h>
#include <fst/extensions/pdt/pdt.h>
#include <fst/shortest-path.h>

namespace fst {

template <class Arc, class Queue>
struct PdtShortestPathOptions {
  bool keep_parentheses;
  bool path_gc;

  PdtShortestPathOptions(bool keep_parentheses = false, bool path_gc = true)
      : keep_parentheses(keep_parentheses), path_gc(path_gc) {}
};

namespace internal {

// Flags for shortest path data.

constexpr uint8_t kPdtInited = 0x01;
constexpr uint8_t kPdtFinal = 0x02;
constexpr uint8_t kPdtMarked = 0x04;

// Stores shortest path tree info Distance(), Parent(), and ArcParent()
// information keyed on two types:
//
// 1. SearchState: This is a usual node in a shortest path tree but:
//    a. is w.r.t a PDT search state (a pair of a PDT state and a "start" state,
//    either the PDT start state or the destination state of an open
//    parenthesis).
//    b. the Distance() is from this "start" state to the search state.
//    c. Parent().state is kNoLabel for the "start" state.
//
// 2. ParenSpec: This connects shortest path trees depending on the the
// parenthesis taken. Given the parenthesis spec:
//    a. the Distance() is from the Parent() "start" state to the parenthesis
//    destination state.
//    b. The ArcParent() is the parenthesis arc.
template <class Arc>
class PdtShortestPathData {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  struct SearchState {
    StateId state;  // PDT state.
    StateId start;  // PDT paren "start" state.

    SearchState(StateId s = kNoStateId, StateId t = kNoStateId)
        : state(s), start(t) {}

    bool operator==(const SearchState &other) const {
      if (&other == this) return true;
      return other.state == state && other.start == start;
    }
  };

  // Specifies paren ID, source and dest "start" states of a paren. These are
  // the "start" states of the respective sub-graphs.
  struct ParenSpec {
    ParenSpec(Label paren_id = kNoLabel, StateId src_start = kNoStateId,
              StateId dest_start = kNoStateId)
        : paren_id(paren_id), src_start(src_start), dest_start(dest_start) {}

    Label paren_id;
    StateId src_start;   // Sub-graph "start" state for paren source.
    StateId dest_start;  // Sub-graph "start" state for paren dest.

    bool operator==(const ParenSpec &other) const {
      if (&other == this) return true;
      return (other.paren_id == paren_id &&
              other.src_start == other.src_start &&
              other.dest_start == dest_start);
    }
  };

  struct SearchData {
    SearchData()
        : distance(Weight::Zero()),
          parent(kNoStateId, kNoStateId),
          paren_id(kNoLabel),
          flags(0) {}

    Weight distance;     // Distance to this state from PDT "start" state.
    SearchState parent;  // Parent state in shortest path tree.
    int16_t paren_id;      // If parent arc has paren, paren ID (or kNoLabel).
    uint8_t flags;         // First byte reserved for PdtShortestPathData use.
  };

  PdtShortestPathData(bool gc)
      : gc_(gc), nstates_(0), ngc_(0), finished_(false) {}

  ~PdtShortestPathData() {
    VLOG(1) << "opm size: " << paren_map_.size();
    VLOG(1) << "# of search states: " << nstates_;
    if (gc_) VLOG(1) << "# of GC'd search states: " << ngc_;
  }

  void Clear() {
    search_map_.clear();
    search_multimap_.clear();
    paren_map_.clear();
    state_ = SearchState(kNoStateId, kNoStateId);
    nstates_ = 0;
    ngc_ = 0;
  }

  // TODO(kbg): Currently copying SearchState and passing a const reference to
  // ParenSpec. Benchmark to confirm this is the right thing to do.

  Weight Distance(SearchState s) const { return GetSearchData(s)->distance; }

  Weight Distance(const ParenSpec &paren) const {
    return GetSearchData(paren)->distance;
  }

  SearchState Parent(SearchState s) const { return GetSearchData(s)->parent; }

  SearchState Parent(const ParenSpec &paren) const {
    return GetSearchData(paren)->parent;
  }

  Label ParenId(SearchState s) const { return GetSearchData(s)->paren_id; }

  uint8_t Flags(SearchState s) const { return GetSearchData(s)->flags; }

  void SetDistance(SearchState s, Weight weight) {
    GetSearchData(s)->distance = std::move(weight);
  }

  void SetDistance(const ParenSpec &paren, Weight weight) {
    GetSearchData(paren)->distance = std::move(weight);
  }

  void SetParent(SearchState s, SearchState p) { GetSearchData(s)->parent = p; }

  void SetParent(const ParenSpec &paren, SearchState p) {
    GetSearchData(paren)->parent = p;
  }

  void SetParenId(SearchState s, Label p) {
    if (p >= 32768) {
      FSTERROR() << "PdtShortestPathData: Paren ID does not fit in an int16_t";
    }
    GetSearchData(s)->paren_id = p;
  }

  void SetFlags(SearchState s, uint8_t f, uint8_t mask) {
    auto *data = GetSearchData(s);
    data->flags &= ~mask;
    data->flags |= f & mask;
  }

  void GC(StateId s);

  void Finish() { finished_ = true; }

 private:
  // Hash for search state.
  struct SearchStateHash {
    size_t operator()(const SearchState &s) const {
      static constexpr auto prime = 7853;
      return s.state + s.start * prime;
    }
  };

  // Hash for paren map.
  struct ParenHash {
    size_t operator()(const ParenSpec &paren) const {
      static constexpr auto prime0 = 7853;
      static constexpr auto prime1 = 7867;
      return paren.paren_id + paren.src_start * prime0 +
             paren.dest_start * prime1;
    }
  };

  using SearchMap =
      std::unordered_map<SearchState, SearchData, SearchStateHash>;

  using SearchMultimap = std::unordered_multimap<StateId, StateId>;

  // Hash map from paren spec to open paren data.
  using ParenMap = std::unordered_map<ParenSpec, SearchData, ParenHash>;

  SearchData *GetSearchData(SearchState s) const {
    if (s == state_) return state_data_;
    if (finished_) {
      auto it = search_map_.find(s);
      if (it == search_map_.end()) return &null_search_data_;
      state_ = s;
      return state_data_ = &(it->second);
    } else {
      state_ = s;
      state_data_ = &search_map_[s];
      if (!(state_data_->flags & kPdtInited)) {
        ++nstates_;
        if (gc_) search_multimap_.insert(std::make_pair(s.start, s.state));
        state_data_->flags = kPdtInited;
      }
      return state_data_;
    }
  }

  SearchData *GetSearchData(ParenSpec paren) const {
    if (paren == paren_) return paren_data_;
    if (finished_) {
      auto it = paren_map_.find(paren);
      if (it == paren_map_.end()) return &null_search_data_;
      paren_ = paren;
      return state_data_ = &(it->second);
    } else {
      paren_ = paren;
      return paren_data_ = &paren_map_[paren];
    }
  }

  mutable SearchMap search_map_;            // Maps from search state to data.
  mutable SearchMultimap search_multimap_;  // Maps from "start" to subgraph.
  mutable ParenMap paren_map_;              // Maps paren spec to search data.
  mutable SearchState state_;               // Last state accessed.
  mutable SearchData *state_data_;          // Last state data accessed.
  mutable ParenSpec paren_;                 // Last paren spec accessed.
  mutable SearchData *paren_data_;          // Last paren data accessed.
  bool gc_;                                 // Allow GC?
  mutable size_t nstates_;                  // Total number of search states.
  size_t ngc_;                              // Number of GC'd search states.
  mutable SearchData null_search_data_;     // Null search data.
  bool finished_;                           // Read-only access when true.

  PdtShortestPathData(const PdtShortestPathData &) = delete;
  PdtShortestPathData &operator=(const PdtShortestPathData &) = delete;
};

// Deletes inaccessible search data from a given "start" (open paren dest)
// state. Assumes "final" (close paren source or PDT final) states have
// been flagged kPdtFinal.
template <class Arc>
void PdtShortestPathData<Arc>::GC(StateId start) {
  if (!gc_) return;
  std::vector<StateId> finals;
  for (auto it = search_multimap_.find(start);
       it != search_multimap_.end() && it->first == start; ++it) {
    const SearchState s(it->second, start);
    if (search_map_[s].flags & kPdtFinal) finals.push_back(s.state);
  }
  // Mark phase.
  for (const auto state : finals) {
    SearchState ss(state, start);
    while (ss.state != kNoLabel) {
      auto &sdata = search_map_[ss];
      if (sdata.flags & kPdtMarked) break;
      sdata.flags |= kPdtMarked;
      const auto p = sdata.parent;
      if (p.start != start && p.start != kNoLabel) {  // Entering sub-subgraph.
        const ParenSpec paren(sdata.paren_id, ss.start, p.start);
        ss = paren_map_[paren].parent;
      } else {
        ss = p;
      }
    }
  }
  // Sweep phase.
  auto it = search_multimap_.find(start);
  while (it != search_multimap_.end() && it->first == start) {
    const SearchState s(it->second, start);
    auto mit = search_map_.find(s);
    const SearchData &data = mit->second;
    if (!(data.flags & kPdtMarked)) {
      search_map_.erase(mit);
      ++ngc_;
    }
    search_multimap_.erase(it++);
  }
}

}  // namespace internal

// This computes the single source shortest (balanced) path (SSSP) through a
// weighted PDT that has a bounded stack (i.e., is expandable as an FST). It is
// a generalization of the classic SSSP graph algorithm that removes a state s
// from a queue (defined by a user-provided queue type) and relaxes the
// destination states of transitions leaving s. In this PDT version, states that
// have entering open parentheses are treated as source states for a sub-graph
// SSSP problem with the shortest path up to the open parenthesis being first
// saved. When a close parenthesis is then encountered any balancing open
// parenthesis is examined for this saved information and multiplied back. In
// this way, each sub-graph is entered only once rather than repeatedly. If
// every state in the input PDT has the property that there is a unique "start"
// state for it with entering open parentheses, then this algorithm is quite
// straightforward. In general, this will not be the case, so the algorithm
// (implicitly) creates a new graph where each state is a pair of an original
// state and a possible parenthesis "start" state for that state.
template <class Arc, class Queue>
class PdtShortestPath {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using SpData = internal::PdtShortestPathData<Arc>;
  using SearchState = typename SpData::SearchState;
  using ParenSpec = typename SpData::ParenSpec;
  using CloseSourceIterator =
      typename internal::PdtBalanceData<Arc>::SetIterator;

  PdtShortestPath(const Fst<Arc> &ifst,
                  const std::vector<std::pair<Label, Label>> &parens,
                  const PdtShortestPathOptions<Arc, Queue> &opts)
      : ifst_(ifst.Copy()),
        parens_(parens),
        keep_parens_(opts.keep_parentheses),
        start_(ifst.Start()),
        sp_data_(opts.path_gc),
        error_(false) {
    // TODO(kbg): Make this a compile-time static_assert once:
    // 1) All weight properties are made constexpr for all weight types.
    // 2) We have a pleasant way to "deregister" this oepration for non-path
    //    semirings so an informative error message is produced. The best
    //    solution will probably involve some kind of SFINAE magic.
    if ((Weight::Properties() & (kPath | kRightSemiring)) !=
        (kPath | kRightSemiring)) {
      FSTERROR() << "PdtShortestPath: Weight needs to have the path"
                 << " property and be right distributive: " << Weight::Type();
      error_ = true;
    }
    for (Label i = 0; i < parens.size(); ++i) {
      const auto &pair = parens[i];
      paren_map_[pair.first] = i;
      paren_map_[pair.second] = i;
    }
  }

  ~PdtShortestPath() {
    VLOG(1) << "# of input states: " << CountStates(*ifst_);
    VLOG(1) << "# of enqueued: " << nenqueued_;
    VLOG(1) << "cpmm size: " << close_paren_multimap_.size();
  }

  void ShortestPath(MutableFst<Arc> *ofst) {
    Init(ofst);
    GetDistance(start_);
    GetPath();
    sp_data_.Finish();
    if (error_) ofst->SetProperties(kError, kError);
  }

  const internal::PdtShortestPathData<Arc> &GetShortestPathData() const {
    return sp_data_;
  }

  internal::PdtBalanceData<Arc> *GetBalanceData() { return &balance_data_; }

 public:
  // Hash multimap from close paren label to an paren arc.
  using CloseParenMultimap =
      std::unordered_multimap<internal::ParenState<Arc>, Arc,
                              typename internal::ParenState<Arc>::Hash>;

  const CloseParenMultimap &GetCloseParenMultimap() const {
    return close_paren_multimap_;
  }

 private:
  void Init(MutableFst<Arc> *ofst);

  void GetDistance(StateId start);

  void ProcFinal(SearchState s);

  void ProcArcs(SearchState s);

  void ProcOpenParen(Label paren_id, SearchState s, StateId nexstate,
                     const Weight &weight);

  void ProcCloseParen(Label paren_id, SearchState s, const Weight &weight);

  void ProcNonParen(SearchState s, StateId nextstate, const Weight &weight);

  void Relax(SearchState s, SearchState t, StateId nextstate,
             const Weight &weight, Label paren_id);

  void Enqueue(SearchState d);

  void GetPath();

  Arc GetPathArc(SearchState s, SearchState p, Label paren_id, bool open);

  std::unique_ptr<Fst<Arc>> ifst_;
  MutableFst<Arc> *ofst_;
  const std::vector<std::pair<Label, Label>> &parens_;
  bool keep_parens_;
  Queue *state_queue_;
  StateId start_;
  Weight fdistance_;
  SearchState f_parent_;
  SpData sp_data_;
  std::unordered_map<Label, Label> paren_map_;
  CloseParenMultimap close_paren_multimap_;
  internal::PdtBalanceData<Arc> balance_data_;
  std::ptrdiff_t nenqueued_;
  bool error_;

  static constexpr uint8_t kEnqueued = 0x10;
  static constexpr uint8_t kExpanded = 0x20;
  static constexpr uint8_t kFinished = 0x40;

  static const Arc kNoArc;
};

template <class Arc, class Queue>
void PdtShortestPath<Arc, Queue>::Init(MutableFst<Arc> *ofst) {
  ofst_ = ofst;
  ofst->DeleteStates();
  ofst->SetInputSymbols(ifst_->InputSymbols());
  ofst->SetOutputSymbols(ifst_->OutputSymbols());
  if (ifst_->Start() == kNoStateId) return;
  fdistance_ = Weight::Zero();
  f_parent_ = SearchState(kNoStateId, kNoStateId);
  sp_data_.Clear();
  close_paren_multimap_.clear();
  balance_data_.Clear();
  nenqueued_ = 0;
  // Finds open parens per destination state and close parens per source state.
  for (StateIterator<Fst<Arc>> siter(*ifst_); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    for (ArcIterator<Fst<Arc>> aiter(*ifst_, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto it = paren_map_.find(arc.ilabel);
      if (it != paren_map_.end()) {  // Is a paren?
        const auto paren_id = it->second;
        if (arc.ilabel == parens_[paren_id].first) {  // Open paren.
          balance_data_.OpenInsert(paren_id, arc.nextstate);
        } else {  // Close paren.
          const internal::ParenState<Arc> paren_state(paren_id, s);
          close_paren_multimap_.emplace(paren_state, arc);
        }
      }
    }
  }
}

// Computes the shortest distance stored in a recursive way. Each sub-graph
// (i.e., different paren "start" state) begins with weight One().
template <class Arc, class Queue>
void PdtShortestPath<Arc, Queue>::GetDistance(StateId start) {
  if (start == kNoStateId) return;
  Queue state_queue;
  state_queue_ = &state_queue;
  const SearchState q(start, start);
  Enqueue(q);
  sp_data_.SetDistance(q, Weight::One());
  while (!state_queue_->Empty()) {
    const auto state = state_queue_->Head();
    state_queue_->Dequeue();
    const SearchState s(state, start);
    sp_data_.SetFlags(s, 0, kEnqueued);
    ProcFinal(s);
    ProcArcs(s);
    sp_data_.SetFlags(s, kExpanded, kExpanded);
  }
  sp_data_.SetFlags(q, kFinished, kFinished);
  balance_data_.FinishInsert(start);
  sp_data_.GC(start);
}

// Updates best complete path.
template <class Arc, class Queue>
void PdtShortestPath<Arc, Queue>::ProcFinal(SearchState s) {
  if (ifst_->Final(s.state) != Weight::Zero() && s.start == start_) {
    const auto weight = Times(sp_data_.Distance(s), ifst_->Final(s.state));
    if (fdistance_ != Plus(fdistance_, weight)) {
      if (f_parent_.state != kNoStateId) {
        sp_data_.SetFlags(f_parent_, 0, internal::kPdtFinal);
      }
      sp_data_.SetFlags(s, internal::kPdtFinal, internal::kPdtFinal);
      fdistance_ = Plus(fdistance_, weight);
      f_parent_ = s;
    }
  }
}

// Processes all arcs leaving the state s.
template <class Arc, class Queue>
void PdtShortestPath<Arc, Queue>::ProcArcs(SearchState s) {
  for (ArcIterator<Fst<Arc>> aiter(*ifst_, s.state); !aiter.Done();
       aiter.Next()) {
    const auto &arc = aiter.Value();
    const auto weight = Times(sp_data_.Distance(s), arc.weight);
    const auto it = paren_map_.find(arc.ilabel);
    if (it != paren_map_.end()) {  // Is a paren?
      const auto paren_id = it->second;
      if (arc.ilabel == parens_[paren_id].first) {
        ProcOpenParen(paren_id, s, arc.nextstate, weight);
      } else {
        ProcCloseParen(paren_id, s, weight);
      }
    } else {
      ProcNonParen(s, arc.nextstate, weight);
    }
  }
}

// Saves the shortest path info for reaching this parenthesis and starts a new
// SSSP in the sub-graph pointed to by the parenthesis if previously unvisited.
// Otherwise it finds any previously encountered closing parentheses and relaxes
// them using the recursively stored shortest distance to them.
template <class Arc, class Queue>
inline void PdtShortestPath<Arc, Queue>::ProcOpenParen(Label paren_id,
                                                       SearchState s,
                                                       StateId nextstate,
                                                       const Weight &weight) {
  const SearchState d(nextstate, nextstate);
  const ParenSpec paren(paren_id, s.start, d.start);
  const auto pdist = sp_data_.Distance(paren);
  if (pdist != Plus(pdist, weight)) {
    sp_data_.SetDistance(paren, weight);
    sp_data_.SetParent(paren, s);
    const auto dist = sp_data_.Distance(d);
    if (dist == Weight::Zero()) {
      auto *state_queue = state_queue_;
      GetDistance(d.start);
      state_queue_ = state_queue;
    } else if (!(sp_data_.Flags(d) & kFinished)) {
      FSTERROR()
          << "PdtShortestPath: open parenthesis recursion: not bounded stack";
      error_ = true;
    }
    for (auto set_iter = balance_data_.Find(paren_id, nextstate);
         !set_iter.Done(); set_iter.Next()) {
      const SearchState cpstate(set_iter.Element(), d.start);
      const internal::ParenState<Arc> paren_state(paren_id, cpstate.state);
      for (auto cpit = close_paren_multimap_.find(paren_state);
           cpit != close_paren_multimap_.end() && paren_state == cpit->first;
           ++cpit) {
        const auto &cparc = cpit->second;
        const auto cpw =
            Times(weight, Times(sp_data_.Distance(cpstate), cparc.weight));
        Relax(cpstate, s, cparc.nextstate, cpw, paren_id);
      }
    }
  }
}

// Saves the correspondence between each closing parenthesis and its balancing
// open parenthesis info. Relaxes any close parenthesis destination state that
// has a balancing previously encountered open parenthesis.
template <class Arc, class Queue>
inline void PdtShortestPath<Arc, Queue>::ProcCloseParen(Label paren_id,
                                                        SearchState s,
                                                        const Weight &weight) {
  const internal::ParenState<Arc> paren_state(paren_id, s.start);
  if (!(sp_data_.Flags(s) & kExpanded)) {
    balance_data_.CloseInsert(paren_id, s.start, s.state);
    sp_data_.SetFlags(s, internal::kPdtFinal, internal::kPdtFinal);
  }
}

// Classical relaxation for non-parentheses.
template <class Arc, class Queue>
inline void PdtShortestPath<Arc, Queue>::ProcNonParen(SearchState s,
                                                      StateId nextstate,
                                                      const Weight &weight) {
  Relax(s, s, nextstate, weight, kNoLabel);
}

// Classical relaxation on the search graph for an arc with destination state
// nexstate from state s. State t is in the same sub-graph as nextstate (i.e.,
// has the same paren "start").
template <class Arc, class Queue>
inline void PdtShortestPath<Arc, Queue>::Relax(SearchState s, SearchState t,
                                               StateId nextstate,
                                               const Weight &weight,
                                               Label paren_id) {
  const SearchState d(nextstate, t.start);
  Weight dist = sp_data_.Distance(d);
  if (dist != Plus(dist, weight)) {
    sp_data_.SetParent(d, s);
    sp_data_.SetParenId(d, paren_id);
    sp_data_.SetDistance(d, Plus(dist, weight));
    Enqueue(d);
  }
}

template <class Arc, class Queue>
inline void PdtShortestPath<Arc, Queue>::Enqueue(SearchState s) {
  if (!(sp_data_.Flags(s) & kEnqueued)) {
    state_queue_->Enqueue(s.state);
    sp_data_.SetFlags(s, kEnqueued, kEnqueued);
    ++nenqueued_;
  } else {
    state_queue_->Update(s.state);
  }
}

// Follows parent pointers to find the shortest path. A stack is used since the
// shortest distance is stored recursively.
template <class Arc, class Queue>
void PdtShortestPath<Arc, Queue>::GetPath() {
  SearchState s = f_parent_;
  SearchState d = SearchState(kNoStateId, kNoStateId);
  StateId s_p = kNoStateId;
  StateId d_p = kNoStateId;
  auto arc = kNoArc;
  Label paren_id = kNoLabel;
  std::stack<ParenSpec> paren_stack;
  while (s.state != kNoStateId) {
    d_p = s_p;
    s_p = ofst_->AddState();
    if (d.state == kNoStateId) {
      ofst_->SetFinal(s_p, ifst_->Final(f_parent_.state));
    } else {
      if (paren_id != kNoLabel) {                     // Paren?
        if (arc.ilabel == parens_[paren_id].first) {  // Open paren?
          paren_stack.pop();
        } else {  // Close paren?
          const ParenSpec paren(paren_id, d.start, s.start);
          paren_stack.push(paren);
        }
        if (!keep_parens_) arc.ilabel = arc.olabel = 0;
      }
      arc.nextstate = d_p;
      ofst_->AddArc(s_p, arc);
    }
    d = s;
    s = sp_data_.Parent(d);
    paren_id = sp_data_.ParenId(d);
    if (s.state != kNoStateId) {
      arc = GetPathArc(s, d, paren_id, false);
    } else if (!paren_stack.empty()) {
      const ParenSpec paren = paren_stack.top();
      s = sp_data_.Parent(paren);
      paren_id = paren.paren_id;
      arc = GetPathArc(s, d, paren_id, true);
    }
  }
  ofst_->SetStart(s_p);
  ofst_->SetProperties(
      ShortestPathProperties(ofst_->Properties(kFstProperties, false)),
      kFstProperties);
}

// Finds transition with least weight between two states with label matching
// paren_id and open/close paren type or a non-paren if kNoLabel.
template <class Arc, class Queue>
Arc PdtShortestPath<Arc, Queue>::GetPathArc(SearchState s, SearchState d,
                                            Label paren_id, bool open_paren) {
  auto path_arc = kNoArc;
  for (ArcIterator<Fst<Arc>> aiter(*ifst_, s.state); !aiter.Done();
       aiter.Next()) {
    const auto &arc = aiter.Value();
    if (arc.nextstate != d.state) continue;
    Label arc_paren_id = kNoLabel;
    const auto it = paren_map_.find(arc.ilabel);
    if (it != paren_map_.end()) {
      arc_paren_id = it->second;
      bool arc_open_paren = (arc.ilabel == parens_[arc_paren_id].first);
      if (arc_open_paren != open_paren) continue;
    }
    if (arc_paren_id != paren_id) continue;
    if (arc.weight == Plus(arc.weight, path_arc.weight)) path_arc = arc;
  }
  if (path_arc.nextstate == kNoStateId) {
    FSTERROR() << "PdtShortestPath::GetPathArc: Failed to find arc";
    error_ = true;
  }
  return path_arc;
}

template <class Arc, class Queue>
const Arc PdtShortestPath<Arc, Queue>::kNoArc = Arc(kNoLabel, kNoLabel,
                                                    Weight::Zero(), kNoStateId);

// Functional variants.

template <class Arc, class Queue>
void ShortestPath(
    const Fst<Arc> &ifst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens,
    MutableFst<Arc> *ofst, const PdtShortestPathOptions<Arc, Queue> &opts) {
  PdtShortestPath<Arc, Queue> psp(ifst, parens, opts);
  psp.ShortestPath(ofst);
}

template <class Arc>
void ShortestPath(
    const Fst<Arc> &ifst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens,
    MutableFst<Arc> *ofst) {
  using Q = FifoQueue<typename Arc::StateId>;
  const PdtShortestPathOptions<Arc, Q> opts;
  PdtShortestPath<Arc, Q> psp(ifst, parens, opts);
  psp.ShortestPath(ofst);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_SHORTEST_PATH_H_
