// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Expands a PDT to an FST.

#ifndef FST_EXTENSIONS_PDT_EXPAND_H_
#define FST_EXTENSIONS_PDT_EXPAND_H_

#include <forward_list>
#include <vector>

#include <fst/log.h>

#include <fst/extensions/pdt/paren.h>
#include <fst/extensions/pdt/pdt.h>
#include <fst/extensions/pdt/reverse.h>
#include <fst/extensions/pdt/shortest-path.h>
#include <fst/cache.h>
#include <fst/mutable-fst.h>
#include <fst/queue.h>
#include <fst/state-table.h>
#include <fst/test-properties.h>

namespace fst {

template <class Arc>
struct PdtExpandFstOptions : public CacheOptions {
  bool keep_parentheses;
  PdtStack<typename Arc::StateId, typename Arc::Label> *stack;
  PdtStateTable<typename Arc::StateId, typename Arc::StateId> *state_table;

  explicit PdtExpandFstOptions(
      const CacheOptions &opts = CacheOptions(), bool keep_parentheses = false,
      PdtStack<typename Arc::StateId, typename Arc::Label> *stack = nullptr,
      PdtStateTable<typename Arc::StateId, typename Arc::StateId> *state_table =
          nullptr)
      : CacheOptions(opts),
        keep_parentheses(keep_parentheses),
        stack(stack),
        state_table(state_table) {}
};

namespace internal {

// Implementation class for PdtExpandFst.
template <class Arc>
class PdtExpandFstImpl : public CacheImpl<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StackId = StateId;
  using StateTuple = PdtStateTuple<StateId, StackId>;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  using CacheBaseImpl<CacheState<Arc>>::PushArc;
  using CacheBaseImpl<CacheState<Arc>>::HasArcs;
  using CacheBaseImpl<CacheState<Arc>>::HasFinal;
  using CacheBaseImpl<CacheState<Arc>>::HasStart;
  using CacheBaseImpl<CacheState<Arc>>::SetArcs;
  using CacheBaseImpl<CacheState<Arc>>::SetFinal;
  using CacheBaseImpl<CacheState<Arc>>::SetStart;

  PdtExpandFstImpl(const Fst<Arc> &fst,
                   const std::vector<std::pair<Label, Label>> &parens,
                   const PdtExpandFstOptions<Arc> &opts)
      : CacheImpl<Arc>(opts),
        fst_(fst.Copy()),
        stack_(opts.stack ? opts.stack : new PdtStack<StateId, Label>(parens)),
        state_table_(opts.state_table ? opts.state_table
                                      : new PdtStateTable<StateId, StackId>()),
        own_stack_(opts.stack == 0),
        own_state_table_(opts.state_table == 0),
        keep_parentheses_(opts.keep_parentheses) {
    SetType("expand");
    const auto props = fst.Properties(kFstProperties, false);
    SetProperties(PdtExpandProperties(props), kCopyProperties);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  PdtExpandFstImpl(const PdtExpandFstImpl &impl)
      : CacheImpl<Arc>(impl),
        fst_(impl.fst_->Copy(true)),
        stack_(new PdtStack<StateId, Label>(*impl.stack_)),
        state_table_(new PdtStateTable<StateId, StackId>()),
        own_stack_(true),
        own_state_table_(true),
        keep_parentheses_(impl.keep_parentheses_) {
    SetType("expand");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  ~PdtExpandFstImpl() override {
    if (own_stack_) delete stack_;
    if (own_state_table_) delete state_table_;
  }

  StateId Start() {
    if (!HasStart()) {
      const auto s = fst_->Start();
      if (s == kNoStateId) return kNoStateId;
      StateTuple tuple(s, 0);
      const auto start = state_table_->FindState(tuple);
      SetStart(start);
    }
    return CacheImpl<Arc>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      const auto &tuple = state_table_->Tuple(s);
      const auto weight = fst_->Final(tuple.state_id);
      if (weight != Weight::Zero() && tuple.stack_id == 0)
        SetFinal(s, weight);
      else
        SetFinal(s, Weight::Zero());
    }
    return CacheImpl<Arc>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) ExpandState(s);
    return CacheImpl<Arc>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) ExpandState(s);
    return CacheImpl<Arc>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) ExpandState(s);
    return CacheImpl<Arc>::NumOutputEpsilons(s);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) ExpandState(s);
    CacheImpl<Arc>::InitArcIterator(s, data);
  }

  // Computes the outgoing transitions from a state, creating new destination
  // states as needed.
  void ExpandState(StateId s) {
    StateTuple tuple = state_table_->Tuple(s);
    for (ArcIterator<Fst<Arc>> aiter(*fst_, tuple.state_id); !aiter.Done();
         aiter.Next()) {
      auto arc = aiter.Value();
      const auto stack_id = stack_->Find(tuple.stack_id, arc.ilabel);
      if (stack_id == -1) {  // Non-matching close parenthesis.
        continue;
      } else if ((stack_id != tuple.stack_id) && !keep_parentheses_) {
        // Stack push/pop.
        arc.ilabel = 0;
        arc.olabel = 0;
      }
      StateTuple ntuple(arc.nextstate, stack_id);
      arc.nextstate = state_table_->FindState(ntuple);
      PushArc(s, arc);
    }
    SetArcs(s);
  }

  const PdtStack<StackId, Label> &GetStack() const { return *stack_; }

  const PdtStateTable<StateId, StackId> &GetStateTable() const {
    return *state_table_;
  }

 private:
  // Properties for an expanded PDT.
  inline uint64_t PdtExpandProperties(uint64_t inprops) {
    return inprops & (kAcceptor | kAcyclic | kInitialAcyclic | kUnweighted);
  }

  std::unique_ptr<const Fst<Arc>> fst_;
  PdtStack<StackId, Label> *stack_;
  PdtStateTable<StateId, StackId> *state_table_;
  bool own_stack_;
  bool own_state_table_;
  bool keep_parentheses_;
};

}  // namespace internal

// Expands a pushdown transducer (PDT) encoded as an FST into an FST. This
// version is a delayed FST. In the PDT, some transitions are labeled with open
// or close parentheses. To be interpreted as a PDT, the parens must balance on
// a path. The open-close parenthesis label pairs are passed using the parens
// argument. The expansion enforces the parenthesis constraints. The PDT must be
// expandable as an FST.
//
// This class attaches interface to implementation and handles reference
// counting, delegating most methods to ImplToFst.
template <class A>
class PdtExpandFst : public ImplToFst<internal::PdtExpandFstImpl<A>> {
 public:
  using Arc = A;

  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StackId = StateId;
  using Store = DefaultCacheStore<Arc>;
  using State = typename Store::State;
  using Impl = internal::PdtExpandFstImpl<Arc>;

  friend class ArcIterator<PdtExpandFst<Arc>>;
  friend class StateIterator<PdtExpandFst<Arc>>;

  PdtExpandFst(const Fst<Arc> &fst,
               const std::vector<std::pair<Label, Label>> &parens)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, parens, PdtExpandFstOptions<A>())) {}

  PdtExpandFst(const Fst<Arc> &fst,
               const std::vector<std::pair<Label, Label>> &parens,
               const PdtExpandFstOptions<Arc> &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, parens, opts)) {}

  // See Fst<>::Copy() for doc.
  PdtExpandFst(const PdtExpandFst<Arc> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Gets a copy of this ExpandFst. See Fst<>::Copy() for further doc.
  PdtExpandFst<Arc> *Copy(bool safe = false) const override {
    return new PdtExpandFst<Arc>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  const PdtStack<StackId, Label> &GetStack() const {
    return GetImpl()->GetStack();
  }

  const PdtStateTable<StateId, StackId> &GetStateTable() const {
    return GetImpl()->GetStateTable();
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  void operator=(const PdtExpandFst &) = delete;
};

// Specialization for PdtExpandFst.
template <class Arc>
class StateIterator<PdtExpandFst<Arc>>
    : public CacheStateIterator<PdtExpandFst<Arc>> {
 public:
  explicit StateIterator(const PdtExpandFst<Arc> &fst)
      : CacheStateIterator<PdtExpandFst<Arc>>(fst, fst.GetMutableImpl()) {}
};

// Specialization for PdtExpandFst.
template <class Arc>
class ArcIterator<PdtExpandFst<Arc>>
    : public CacheArcIterator<PdtExpandFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const PdtExpandFst<Arc> &fst, StateId s)
      : CacheArcIterator<PdtExpandFst<Arc>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->ExpandState(s);
  }
};

template <class Arc>
inline void PdtExpandFst<Arc>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<PdtExpandFst<Arc>>(*this);
}

// PrunedExpand prunes the delayed expansion of a pushdown transducer (PDT)
// encoded as an FST into an FST. In the PDT, some transitions are labeled with
// open or close parentheses. To be interpreted as a PDT, the parens must
// balance on a path. The open-close parenthesis label pairs are passed
// using the parens argument. The expansion enforces the parenthesis
// constraints.
//
// The algorithm works by visiting the delayed ExpandFst using a shortest-stack
// first queue discipline and relies on the shortest-distance information
// computed using a reverse shortest-path call to perform the pruning.
//
// The algorithm maintains the same state ordering between the ExpandFst being
// visited (efst_) and the result of pruning written into the MutableFst (ofst_)
// to improve readability.
template <class Arc>
class PdtPrunedExpand {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StackId = StateId;
  using Stack = PdtStack<StackId, Label>;
  using StateTable = PdtStateTable<StateId, StackId>;
  using SetIterator = typename internal::PdtBalanceData<Arc>::SetIterator;

  // Constructor taking as input a PDT specified by by an input FST and a vector
  // of parentheses. The keep_parentheses argument specifies whether parentheses
  // are replaced by epsilons or not during the expansion. The cache options are
  // passed to the underlying ExpandFst.
  PdtPrunedExpand(const Fst<Arc> &ifst,
                  const std::vector<std::pair<Label, Label>> &parens,
                  bool keep_parentheses = false,
                  const CacheOptions &opts = CacheOptions())
      : ifst_(ifst.Copy()),
        keep_parentheses_(keep_parentheses),
        stack_(parens),
        efst_(ifst, parens,
              PdtExpandFstOptions<Arc>(opts, true, &stack_, &state_table_)),
        queue_(state_table_, stack_, stack_length_, distance_, fdistance_),
        error_(false) {
    Reverse(*ifst_, parens, &rfst_);
    VectorFst<Arc> path;
    reverse_shortest_path_.reset(new PdtShortestPath<Arc, FifoQueue<StateId>>(
        rfst_, parens,
        PdtShortestPathOptions<Arc, FifoQueue<StateId>>(true, false)));
    reverse_shortest_path_->ShortestPath(&path);
    error_ = (path.Properties(kError, true) == kError);
    balance_data_.reset(reverse_shortest_path_->GetBalanceData()->Reverse(
        rfst_.NumStates(), 10, -1));
    InitCloseParenMultimap(parens);
  }

  bool Error() const { return error_; }

  // Expands and prunes the input PDT according to the provided weight
  // threshold, wirting the result into an output mutable FST.
  void Expand(MutableFst<Arc> *ofst, const Weight &threshold);

 private:
  static constexpr uint8_t kEnqueued = 0x01;
  static constexpr uint8_t kExpanded = 0x02;
  static constexpr uint8_t kSourceState = 0x04;

  // Comparison functor used by the queue:
  //
  // 1. States corresponding to shortest stack first, and
  // 2. for stacks of matching length, reverse lexicographic order is used, and
  // 3. for states with the same stack, shortest-first order is used.
  class StackCompare {
   public:
    StackCompare(const StateTable &state_table, const Stack &stack,
                 const std::vector<StackId> &stack_length,
                 const std::vector<Weight> &distance,
                 const std::vector<Weight> &fdistance)
        : state_table_(state_table),
          stack_(stack),
          stack_length_(stack_length),
          distance_(distance),
          fdistance_(fdistance) {}

    bool operator()(StateId s1, StateId s2) const {
      auto si1 = state_table_.Tuple(s1).stack_id;
      auto si2 = state_table_.Tuple(s2).stack_id;
      if (stack_length_[si1] < stack_length_[si2]) return true;
      if (stack_length_[si1] > stack_length_[si2]) return false;
      // If stack IDs are equal, use A*.
      if (si1 == si2) {
        return less_(Distance(s1), Distance(s2));
      }
      // If lengths are equal, uses reverse lexicographic order.
      for (; si1 != si2; si1 = stack_.Pop(si1), si2 = stack_.Pop(si2)) {
        if (stack_.Top(si1) < stack_.Top(si2)) return true;
        if (stack_.Top(si1) > stack_.Top(si2)) return false;
      }
      return false;
    }

   private:
    Weight Distance(StateId s) const {
      return (s < distance_.size()) && (s < fdistance_.size())
                 ? Times(distance_[s], fdistance_[s])
                 : Weight::Zero();
    }

    const StateTable &state_table_;
    const Stack &stack_;
    const std::vector<StackId> &stack_length_;
    const std::vector<Weight> &distance_;
    const std::vector<Weight> &fdistance_;
    const NaturalLess<Weight> less_;
  };

  class ShortestStackFirstQueue
      : public ShortestFirstQueue<StateId, StackCompare> {
   public:
    ShortestStackFirstQueue(const PdtStateTable<StateId, StackId> &state_table,
                            const Stack &stack,
                            const std::vector<StackId> &stack_length,
                            const std::vector<Weight> &distance,
                            const std::vector<Weight> &fdistance)
        : ShortestFirstQueue<StateId, StackCompare>(StackCompare(
              state_table, stack, stack_length, distance, fdistance)) {}
  };

  void InitCloseParenMultimap(
      const std::vector<std::pair<Label, Label>> &parens);

  Weight DistanceToDest(StateId source, StateId dest) const;

  uint8_t Flags(StateId s) const;

  void SetFlags(StateId s, uint8_t flags, uint8_t mask);

  Weight Distance(StateId s) const;

  void SetDistance(StateId s, Weight weight);

  Weight FinalDistance(StateId s) const;

  void SetFinalDistance(StateId s, Weight weight);

  StateId SourceState(StateId s) const;

  void SetSourceState(StateId s, StateId p);

  void AddStateAndEnqueue(StateId s);

  void Relax(StateId s, const Arc &arc, Weight weight);

  bool PruneArc(StateId s, const Arc &arc);

  void ProcStart();

  void ProcFinal(StateId s);

  bool ProcNonParen(StateId s, const Arc &arc, bool add_arc);

  bool ProcOpenParen(StateId s, const Arc &arc, StackId si, StackId nsi);

  bool ProcCloseParen(StateId s, const Arc &arc);

  void ProcDestStates(StateId s, StackId si);

  // Input PDT.
  std::unique_ptr<Fst<Arc>> ifst_;
  // Reversed PDT.
  VectorFst<Arc> rfst_;
  // Keep parentheses in ofst?
  const bool keep_parentheses_;
  // State table for efst_.
  StateTable state_table_;
  // Stack trie.
  Stack stack_;
  // Expanded PDT.
  PdtExpandFst<Arc> efst_;
  // Length of stack for given stack ID.
  std::vector<StackId> stack_length_;
  // Distance from initial state in efst_/ofst.
  std::vector<Weight> distance_;
  // Distance to final states in efst_/ofst.
  std::vector<Weight> fdistance_;
  // Queue used to visit efst_.
  ShortestStackFirstQueue queue_;
  // Construction time failure?
  bool error_;
  // Status flags for states in efst_/ofst.
  std::vector<uint8_t> flags_;
  // PDT source state for each expanded state.
  std::vector<StateId> sources_;
  // Shortest path for rfst_.
  std::unique_ptr<PdtShortestPath<Arc, FifoQueue<StateId>>>
      reverse_shortest_path_;
  std::unique_ptr<internal::PdtBalanceData<Arc>> balance_data_;
  // Maps open paren arcs to balancing close paren arcs.
  typename PdtShortestPath<Arc, FifoQueue<StateId>>::CloseParenMultimap
      close_paren_multimap_;
  MutableFst<Arc> *ofst_;  // Output FST.
  Weight limit_;           // Weight limit.

  // Maps a state s in ifst (i.e., the source of a closed paranthesis matching
  // the top of current_stack_id_ to final states in efst_.
  std::unordered_map<StateId, Weight> dest_map_;
  // Stack ID of the states currently at the top of the queue, i.e., the states
  // currently being popped and processed.
  StackId current_stack_id_;
  std::ptrdiff_t current_paren_id_;  // Paren ID at top of current stack.
  std::ptrdiff_t cached_stack_id_;
  StateId cached_source_;
  // The set of pairs of destination states and weights to final states for the
  // source state cached_source_ and the paren ID cached_paren_id_; i.e., the
  // set of source states of a closed parenthesis with paren ID cached_paren_id
  // balancing an incoming open parenthesis with paren ID cached_paren_id_ in
  // state cached_source_.
  std::forward_list<std::pair<StateId, Weight>> cached_dest_list_;
  NaturalLess<Weight> less_;
};

// Initializes close paren multimap, mapping pairs (s, paren_id) to all the arcs
// out of s labeled with close parenthese for paren_id.
template <class Arc>
void PdtPrunedExpand<Arc>::InitCloseParenMultimap(
    const std::vector<std::pair<Label, Label>> &parens) {
  std::unordered_map<Label, Label> paren_map;
  for (size_t i = 0; i < parens.size(); ++i) {
    const auto &pair = parens[i];
    paren_map[pair.first] = i;
    paren_map[pair.second] = i;
  }
  for (StateIterator<Fst<Arc>> siter(*ifst_); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    for (ArcIterator<Fst<Arc>> aiter(*ifst_, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto it = paren_map.find(arc.ilabel);
      if (it == paren_map.end()) continue;
      if (arc.ilabel == parens[it->second].second) {  // Close paren.
        const internal::ParenState<Arc> key(it->second, s);
        close_paren_multimap_.emplace(key, arc);
      }
    }
  }
}

// Returns the weight of the shortest balanced path from source to dest
// in ifst_; dest must be the source state of a close paren arc.
template <class Arc>
typename Arc::Weight PdtPrunedExpand<Arc>::DistanceToDest(StateId source,
                                                          StateId dest) const {
  using SearchState =
      typename PdtShortestPath<Arc, FifoQueue<StateId>>::SearchState;
  const SearchState ss(source + 1, dest + 1);
  const auto distance =
      reverse_shortest_path_->GetShortestPathData().Distance(ss);
  VLOG(2) << "D(" << source << ", " << dest << ") =" << distance;
  return distance;
}

// Returns the flags for state s in ofst_.
template <class Arc>
uint8_t PdtPrunedExpand<Arc>::Flags(StateId s) const {
  return s < flags_.size() ? flags_[s] : 0;
}

// Modifies the flags for state s in ofst_.
template <class Arc>
void PdtPrunedExpand<Arc>::SetFlags(StateId s, uint8_t flags, uint8_t mask) {
  while (flags_.size() <= s) flags_.push_back(0);
  flags_[s] &= ~mask;
  flags_[s] |= flags & mask;
}

// Returns the shortest distance from the initial state to s in ofst_.
template <class Arc>
typename Arc::Weight PdtPrunedExpand<Arc>::Distance(StateId s) const {
  return s < distance_.size() ? distance_[s] : Weight::Zero();
}

// Sets the shortest distance from the initial state to s in ofst_.
template <class Arc>
void PdtPrunedExpand<Arc>::SetDistance(StateId s, Weight weight) {
  while (distance_.size() <= s) distance_.push_back(Weight::Zero());
  distance_[s] = std::move(weight);
}

// Returns the shortest distance from s to the final states in ofst_.
template <class Arc>
typename Arc::Weight PdtPrunedExpand<Arc>::FinalDistance(StateId s) const {
  return s < fdistance_.size() ? fdistance_[s] : Weight::Zero();
}

// Sets the shortest distance from s to the final states in ofst_.
template <class Arc>
void PdtPrunedExpand<Arc>::SetFinalDistance(StateId s, Weight weight) {
  while (fdistance_.size() <= s) fdistance_.push_back(Weight::Zero());
  fdistance_[s] = std::move(weight);
}

// Returns the PDT source state of state s in ofst_.
template <class Arc>
typename Arc::StateId PdtPrunedExpand<Arc>::SourceState(StateId s) const {
  return s < sources_.size() ? sources_[s] : kNoStateId;
}

// Sets the PDT source state of state s in ofst_ to state p'in ifst_.
template <class Arc>
void PdtPrunedExpand<Arc>::SetSourceState(StateId s, StateId p) {
  while (sources_.size() <= s) sources_.push_back(kNoStateId);
  sources_[s] = p;
}

// Adds state s of efst_ to ofst_ and inserts it in the queue, modifying the
// flags for s accordingly.
template <class Arc>
void PdtPrunedExpand<Arc>::AddStateAndEnqueue(StateId s) {
  if (!(Flags(s) & (kEnqueued | kExpanded))) {
    while (ofst_->NumStates() <= s) ofst_->AddState();
    queue_.Enqueue(s);
    SetFlags(s, kEnqueued, kEnqueued);
  } else if (Flags(s) & kEnqueued) {
    queue_.Update(s);
  }
  // TODO(allauzen): Check everything is fine when kExpanded?
}

// Relaxes arc out of state s in ofst_ as follows:
//
// 1. If the distance to s times the weight of arc is smaller than
//   the currently stored distance for arc.nextstate, updates
//   Distance(arc.nextstate) with a new estimate
// 2. If fd is less than the currently stored distance from arc.nextstate to the
// final state, updates with new estimate.
template <class Arc>
void PdtPrunedExpand<Arc>::Relax(StateId s, const Arc &arc, Weight fd) {
  const auto nd = Times(Distance(s), arc.weight);
  if (less_(nd, Distance(arc.nextstate))) {
    SetDistance(arc.nextstate, nd);
    SetSourceState(arc.nextstate, SourceState(s));
  }
  if (less_(fd, FinalDistance(arc.nextstate))) {
    SetFinalDistance(arc.nextstate, fd);
  }
  VLOG(2) << "Relax: " << s << ", d[s] = " << Distance(s) << ", to "
          << arc.nextstate << ", d[ns] = " << Distance(arc.nextstate)
          << ", nd = " << nd;
}

// Returns whether the arc out of state s in efst needs pruned.
template <class Arc>
bool PdtPrunedExpand<Arc>::PruneArc(StateId s, const Arc &arc) {
  VLOG(2) << "Prune ?";
  auto fd = Weight::Zero();
  if ((cached_source_ != SourceState(s)) ||
      (cached_stack_id_ != current_stack_id_)) {
    cached_source_ = SourceState(s);
    cached_stack_id_ = current_stack_id_;
    cached_dest_list_.clear();
    if (cached_source_ != ifst_->Start()) {
      for (auto set_iter =
               balance_data_->Find(current_paren_id_, cached_source_);
           !set_iter.Done(); set_iter.Next()) {
        auto dest = set_iter.Element();
        const auto it = dest_map_.find(dest);
        cached_dest_list_.push_front(*it);
      }
    } else {
      // TODO(allauzen): queue discipline should prevent this from ever
      // happening.
      // Replace by a check.
      cached_dest_list_.push_front(
          std::make_pair(rfst_.Start() - 1, Weight::One()));
    }
  }
  for (auto it = cached_dest_list_.begin(); it != cached_dest_list_.end();
       ++it) {
    const auto d =
        DistanceToDest(state_table_.Tuple(arc.nextstate).state_id, it->first);
    fd = Plus(fd, Times(d, it->second));
  }
  Relax(s, arc, fd);
  return less_(limit_, Times(Distance(s), Times(arc.weight, fd)));
}

// Adds start state of efst_ to ofst_, enqueues it, and initializes the distance
// data structures.
template <class Arc>
void PdtPrunedExpand<Arc>::ProcStart() {
  const auto s = efst_.Start();
  AddStateAndEnqueue(s);
  ofst_->SetStart(s);
  SetSourceState(s, ifst_->Start());
  current_stack_id_ = 0;
  current_paren_id_ = -1;
  stack_length_.push_back(0);
  const auto r = rfst_.Start() - 1;
  cached_source_ = ifst_->Start();
  cached_stack_id_ = 0;
  cached_dest_list_.push_front(std::make_pair(r, Weight::One()));
  const PdtStateTuple<StateId, StackId> tuple(r, 0);
  SetFinalDistance(state_table_.FindState(tuple), Weight::One());
  SetDistance(s, Weight::One());
  const auto d = DistanceToDest(ifst_->Start(), r);
  SetFinalDistance(s, d);
  VLOG(2) << d;
}

// Makes s final in ofst_ if shortest accepting path ending in s is below
// threshold.
template <class Arc>
void PdtPrunedExpand<Arc>::ProcFinal(StateId s) {
  const auto weight = efst_.Final(s);
  if (weight == Weight::Zero()) return;
  if (less_(limit_, Times(Distance(s), weight))) return;
  ofst_->SetFinal(s, weight);
}

// Returns true when an arc (or meta-arc) leaving state s in efst_ is below the
// threshold. When add_arc is true, arc is added to ofst_.
template <class Arc>
bool PdtPrunedExpand<Arc>::ProcNonParen(StateId s, const Arc &arc,
                                        bool add_arc) {
  VLOG(2) << "ProcNonParen: " << s << " to " << arc.nextstate << ", "
          << arc.ilabel << ":" << arc.olabel << " / " << arc.weight
          << ", add_arc = " << (add_arc ? "true" : "false");
  if (PruneArc(s, arc)) return false;
  if (add_arc) ofst_->AddArc(s, arc);
  AddStateAndEnqueue(arc.nextstate);
  return true;
}

// Processes an open paren arc leaving state s in ofst_. When the arc is labeled
// with an open paren,
//
// 1. Considers each (shortest) balanced path starting in s by taking the arc
// and ending by a close paren balancing the open paren of as a meta-arc,
// processing and pruning each meta-arc as a non-paren arc, inserting its
// destination to the queue;
// 2. if at least one of these meta-arcs has not been pruned, adds the
// destination of arc to ofst_ as a new source state for the stack ID nsi, and
// inserts it in the queue.
template <class Arc>
bool PdtPrunedExpand<Arc>::ProcOpenParen(StateId s, const Arc &arc, StackId si,
                                         StackId nsi) {
  // Updates the stack length when needed.
  while (stack_length_.size() <= nsi) stack_length_.push_back(-1);
  if (stack_length_[nsi] == -1) stack_length_[nsi] = stack_length_[si] + 1;
  const auto ns = arc.nextstate;
  VLOG(2) << "Open paren: " << s << "(" << state_table_.Tuple(s).state_id
          << ") to " << ns << "(" << state_table_.Tuple(ns).state_id << ")";
  bool proc_arc = false;
  auto fd = Weight::Zero();
  const auto paren_id = stack_.ParenId(arc.ilabel);
  std::forward_list<StateId> sources;
  for (auto set_iter =
           balance_data_->Find(paren_id, state_table_.Tuple(ns).state_id);
       !set_iter.Done(); set_iter.Next()) {
    sources.push_front(set_iter.Element());
  }
  for (const auto source : sources) {
    VLOG(2) << "Close paren source: " << source;
    const internal::ParenState<Arc> paren_state(paren_id, source);
    for (auto it = close_paren_multimap_.find(paren_state);
         it != close_paren_multimap_.end() && paren_state == it->first; ++it) {
      auto meta_arc = it->second;
      const PdtStateTuple<StateId, StackId> tuple(meta_arc.nextstate, si);
      meta_arc.nextstate = state_table_.FindState(tuple);
      const auto state_id = state_table_.Tuple(ns).state_id;
      const auto d = DistanceToDest(state_id, source);
      VLOG(2) << state_id << ", " << source;
      VLOG(2) << "Meta arc weight = " << arc.weight << " Times " << d
              << " Times " << meta_arc.weight;
      meta_arc.weight = Times(arc.weight, Times(d, meta_arc.weight));
      proc_arc |= ProcNonParen(s, meta_arc, false);
      fd = Plus(
          fd,
          Times(Times(DistanceToDest(state_table_.Tuple(ns).state_id, source),
                      it->second.weight),
                FinalDistance(meta_arc.nextstate)));
    }
  }
  if (proc_arc) {
    VLOG(2) << "Proc open paren " << s << " to " << arc.nextstate;
    ofst_->AddArc(
        s, keep_parentheses_ ? arc : Arc(0, 0, arc.weight, arc.nextstate));
    AddStateAndEnqueue(arc.nextstate);
    const auto nd = Times(Distance(s), arc.weight);
    if (less_(nd, Distance(arc.nextstate))) SetDistance(arc.nextstate, nd);
    // FinalDistance not necessary for source state since pruning decided using
    // meta-arcs above.  But this is a problem with A*, hence the following.
    if (less_(fd, FinalDistance(arc.nextstate)))
      SetFinalDistance(arc.nextstate, fd);
    SetFlags(arc.nextstate, kSourceState, kSourceState);
  }
  return proc_arc;
}

// Checks that shortest path through close paren arc in efst_ is below
// threshold, and if so, adds it to ofst_.
template <class Arc>
bool PdtPrunedExpand<Arc>::ProcCloseParen(StateId s, const Arc &arc) {
  const auto weight =
      Times(Distance(s), Times(arc.weight, FinalDistance(arc.nextstate)));
  if (less_(limit_, weight)) return false;
  ofst_->AddArc(s,
                keep_parentheses_ ? arc : Arc(0, 0, arc.weight, arc.nextstate));
  return true;
}

// When state s in ofst_ is a source state for stack ID si, identifies all the
// corresponding possible destination states, that is, all the states in ifst_
// that have an outgoing close paren arc balancing the incoming open paren taken
// to get to s. For each such state t, computes the shortest distance from (t,
// si) to the final states in ofst_. Stores this information in dest_map_.
template <class Arc>
void PdtPrunedExpand<Arc>::ProcDestStates(StateId s, StackId si) {
  if (!(Flags(s) & kSourceState)) return;
  if (si != current_stack_id_) {
    dest_map_.clear();
    current_stack_id_ = si;
    current_paren_id_ = stack_.Top(current_stack_id_);
    VLOG(2) << "StackID " << si << " dequeued for first time";
  }
  // TODO(allauzen): clean up source state business; rename current function to
  // ProcSourceState.
  SetSourceState(s, state_table_.Tuple(s).state_id);
  const auto paren_id = stack_.Top(si);
  for (auto set_iter =
           balance_data_->Find(paren_id, state_table_.Tuple(s).state_id);
       !set_iter.Done(); set_iter.Next()) {
    const auto dest_state = set_iter.Element();
    if (dest_map_.find(dest_state) != dest_map_.end()) continue;
    auto dest_weight = Weight::Zero();
    internal::ParenState<Arc> paren_state(paren_id, dest_state);
    for (auto it = close_paren_multimap_.find(paren_state);
         it != close_paren_multimap_.end() && paren_state == it->first; ++it) {
      const auto &arc = it->second;
      const PdtStateTuple<StateId, StackId> tuple(arc.nextstate,
                                                  stack_.Pop(si));
      dest_weight =
          Plus(dest_weight,
               Times(arc.weight, FinalDistance(state_table_.FindState(tuple))));
    }
    dest_map_[dest_state] = dest_weight;
    VLOG(2) << "State " << dest_state << " is a dest state for stack ID " << si
            << " with weight " << dest_weight;
  }
}

// Expands and prunes the input PDT, writing the result in ofst.
template <class Arc>
void PdtPrunedExpand<Arc>::Expand(MutableFst<Arc> *ofst,
                                  const typename Arc::Weight &threshold) {
  ofst_ = ofst;
  if (error_) {
    ofst_->SetProperties(kError, kError);
    return;
  }
  ofst_->DeleteStates();
  ofst_->SetInputSymbols(ifst_->InputSymbols());
  ofst_->SetOutputSymbols(ifst_->OutputSymbols());
  limit_ = Times(DistanceToDest(ifst_->Start(), rfst_.Start() - 1), threshold);
  flags_.clear();
  ProcStart();
  while (!queue_.Empty()) {
    const auto s = queue_.Head();
    queue_.Dequeue();
    SetFlags(s, kExpanded, kExpanded | kEnqueued);
    VLOG(2) << s << " dequeued!";
    ProcFinal(s);
    StackId stack_id = state_table_.Tuple(s).stack_id;
    ProcDestStates(s, stack_id);
    for (ArcIterator<PdtExpandFst<Arc>> aiter(efst_, s); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto nextstack_id = state_table_.Tuple(arc.nextstate).stack_id;
      if (stack_id == nextstack_id) {
        ProcNonParen(s, arc, true);
      } else if (stack_id == stack_.Pop(nextstack_id)) {
        ProcOpenParen(s, arc, stack_id, nextstack_id);
      } else {
        ProcCloseParen(s, arc);
      }
    }
    VLOG(2) << "d[" << s << "] = " << Distance(s) << ", fd[" << s
            << "] = " << FinalDistance(s);
  }
}

// Expand functions.

template <class Arc>
struct PdtExpandOptions {
  using Weight = typename Arc::Weight;

  bool connect;
  bool keep_parentheses;
  Weight weight_threshold;

  PdtExpandOptions(bool connect = true, bool keep_parentheses = false,
                   Weight weight_threshold = Weight::Zero())
      : connect(connect),
        keep_parentheses(keep_parentheses),
        weight_threshold(std::move(weight_threshold)) {}
};

// Expands a pushdown transducer (PDT) encoded as an FST into an FST. This
// version writes the expanded PDT to a mutable FST. In the PDT, some
// transitions are labeled with open or close parentheses. To be interpreted as
// a PDT, the parens must balance on a path. The open-close parenthesis label
// pairs are passed using the parens argument. Expansion enforces the
// parenthesis constraints. The PDT must be expandable as an FST.
template <class Arc>
void Expand(
    const Fst<Arc> &ifst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens,
    MutableFst<Arc> *ofst, const PdtExpandOptions<Arc> &opts) {
  PdtExpandFstOptions<Arc> eopts;
  eopts.gc_limit = 0;
  if (opts.weight_threshold == Arc::Weight::Zero()) {
    eopts.keep_parentheses = opts.keep_parentheses;
    *ofst = PdtExpandFst<Arc>(ifst, parens, eopts);
  } else {
    PdtPrunedExpand<Arc> pruned_expand(ifst, parens, opts.keep_parentheses);
    pruned_expand.Expand(ofst, opts.weight_threshold);
  }
  if (opts.connect) Connect(ofst);
}

// Expands a pushdown transducer (PDT) encoded as an FST into an FST. This
// version writes the expanded PDT result to a mutable FST. In the PDT, some
// transitions are labeled with open or close parentheses. To be interpreted as
// a PDT, the parens must balance on a path. The open-close parenthesis label
// pairs are passed using the parents argument. Expansion enforces the
// parenthesis constraints. The PDT must be expandable as an FST.
template <class Arc>
void Expand(const Fst<Arc> &ifst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
    &parens, MutableFst<Arc> *ofst, bool connect = true,
    bool keep_parentheses = false) {
  const PdtExpandOptions<Arc> opts(connect, keep_parentheses);
  Expand(ifst, parens, ofst, opts);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_EXPAND_H_
