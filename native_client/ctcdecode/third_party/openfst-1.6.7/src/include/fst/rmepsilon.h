// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes that implemement epsilon-removal.

#ifndef FST_RMEPSILON_H_
#define FST_RMEPSILON_H_

#include <forward_list>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/arcfilter.h>
#include <fst/cache.h>
#include <fst/connect.h>
#include <fst/factor-weight.h>
#include <fst/invert.h>
#include <fst/prune.h>
#include <fst/queue.h>
#include <fst/shortest-distance.h>
#include <fst/topsort.h>


namespace fst {

template <class Arc, class Queue>
class RmEpsilonOptions
    : public ShortestDistanceOptions<Arc, Queue, EpsilonArcFilter<Arc>> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  bool connect;             // Connect output
  Weight weight_threshold;  // Pruning weight threshold.
  StateId state_threshold;  // Pruning state threshold.

  explicit RmEpsilonOptions(Queue *queue, float delta = kShortestDelta,
                            bool connect = true,
                            Weight weight_threshold = Weight::Zero(),
                            StateId state_threshold = kNoStateId)
      : ShortestDistanceOptions<Arc, Queue, EpsilonArcFilter<Arc>>(
            queue, EpsilonArcFilter<Arc>(), kNoStateId, delta),
        connect(connect),
        weight_threshold(std::move(weight_threshold)),
        state_threshold(state_threshold) {}
};

namespace internal {

// Computation state of the epsilon-removal algorithm.
template <class Arc, class Queue>
class RmEpsilonState {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  RmEpsilonState(const Fst<Arc> &fst, std::vector<Weight> *distance,
                 const RmEpsilonOptions<Arc, Queue> &opts)
      : fst_(fst),
        distance_(distance),
        sd_state_(fst_, distance, opts, true),
        expand_id_(0) {}

  void Expand(StateId s);

  std::vector<Arc> &Arcs() { return arcs_; }

  const Weight &Final() const { return final_; }

  bool Error() const { return sd_state_.Error(); }

 private:
  struct Element {
    Label ilabel;
    Label olabel;
    StateId nextstate;

    Element() {}

    Element(Label ilabel, Label olabel, StateId nexstate)
        : ilabel(ilabel), olabel(olabel), nextstate(nexstate) {}
  };

  struct ElementHash {
   public:
    size_t operator()(const Element &element) const {
      static constexpr size_t prime0 = 7853;
      static constexpr size_t prime1 = 7867;
      return static_cast<size_t>(element.nextstate) +
             static_cast<size_t>(element.ilabel) * prime0 +
             static_cast<size_t>(element.olabel) * prime1;
    }
  };

  class ElementEqual {
   public:
    bool operator()(const Element &e1, const Element &e2) const {
      return (e1.ilabel == e2.ilabel) && (e1.olabel == e2.olabel) &&
             (e1.nextstate == e2.nextstate);
    }
  };

  using ElementMap = std::unordered_map<Element, std::pair<StateId, size_t>,
                                        ElementHash, ElementEqual>;

  const Fst<Arc> &fst_;
  // Distance from state being expanded in epsilon-closure.
  std::vector<Weight> *distance_;
  // Shortest distance algorithm computation state.
  internal::ShortestDistanceState<Arc, Queue, EpsilonArcFilter<Arc>> sd_state_;
  // Maps an element to a pair corresponding to a position in the arcs vector
  // of the state being expanded. The element corresopnds to the position in
  // the arcs_ vector if p.first is equal to the state being expanded.
  ElementMap element_map_;
  EpsilonArcFilter<Arc> eps_filter_;
  std::stack<StateId> eps_queue_;  // Queue used to visit the epsilon-closure.
  std::vector<bool> visited_;      // True if the state has been visited.
  std::forward_list<StateId> visited_states_;  // List of visited states.
  std::vector<Arc> arcs_;                      // Arcs of state being expanded.
  Weight final_;       // Final weight of state being expanded.
  StateId expand_id_;  // Unique ID for each call to Expand

  RmEpsilonState(const RmEpsilonState &) = delete;
  RmEpsilonState &operator=(const RmEpsilonState &) = delete;
};

template <class Arc, class Queue>
void RmEpsilonState<Arc, Queue>::Expand(typename Arc::StateId source) {
  final_ = Weight::Zero();
  arcs_.clear();
  sd_state_.ShortestDistance(source);
  if (sd_state_.Error()) return;
  eps_queue_.push(source);
  while (!eps_queue_.empty()) {
    const auto state = eps_queue_.top();
    eps_queue_.pop();
    while (visited_.size() <= state) visited_.push_back(false);
    if (visited_[state]) continue;
    visited_[state] = true;
    visited_states_.push_front(state);
    for (ArcIterator<Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      auto arc = aiter.Value();
      arc.weight = Times((*distance_)[state], arc.weight);
      if (eps_filter_(arc)) {
        while (visited_.size() <= arc.nextstate) visited_.push_back(false);
        if (!visited_[arc.nextstate]) eps_queue_.push(arc.nextstate);
      } else {
        const Element element(arc.ilabel, arc.olabel, arc.nextstate);
        auto insert_result = element_map_.insert(
            std::make_pair(element, std::make_pair(expand_id_, arcs_.size())));
        if (insert_result.second) {
          arcs_.push_back(arc);
        } else {
          if (insert_result.first->second.first == expand_id_) {
            auto &weight = arcs_[insert_result.first->second.second].weight;
            weight = Plus(weight, arc.weight);
          } else {
            insert_result.first->second.first = expand_id_;
            insert_result.first->second.second = arcs_.size();
            arcs_.push_back(arc);
          }
        }
      }
    }
    final_ = Plus(final_, Times((*distance_)[state], fst_.Final(state)));
  }
  while (!visited_states_.empty()) {
    visited_[visited_states_.front()] = false;
    visited_states_.pop_front();
  }
  ++expand_id_;
}

}  // namespace internal

// Removes epsilon-transitions (when both the input and output label are an
// epsilon) from a transducer. The result will be an equivalent FST that has no
// such epsilon transitions. This version modifies its input. It allows fine
// control via the options argument; see below for a simpler interface.
//
// The distance vector will be used to hold the shortest distances during the
// epsilon-closure computation. The state queue discipline and convergence delta
// are taken in the options argument.
template <class Arc, class Queue>
void RmEpsilon(MutableFst<Arc> *fst,
               std::vector<typename Arc::Weight> *distance,
               const RmEpsilonOptions<Arc, Queue> &opts) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  if (fst->Start() == kNoStateId) return;
  // noneps_in[s] will be set to true iff s admits a non-epsilon incoming
  // transition or is the start state.
  std::vector<bool> noneps_in(fst->NumStates(), false);
  noneps_in[fst->Start()] = true;
  for (size_t i = 0; i < fst->NumStates(); ++i) {
    for (ArcIterator<Fst<Arc>> aiter(*fst, i); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      if (arc.ilabel != 0 || arc.olabel != 0) {
        noneps_in[arc.nextstate] = true;
      }
    }
  }
  // States sorted in topological order when (acyclic) or generic topological
  // order (cyclic).
  std::vector<StateId> states;
  states.reserve(fst->NumStates());
  if (fst->Properties(kTopSorted, false) & kTopSorted) {
    for (size_t i = 0; i < fst->NumStates(); i++) states.push_back(i);
  } else if (fst->Properties(kAcyclic, false) & kAcyclic) {
    std::vector<StateId> order;
    bool acyclic;
    TopOrderVisitor<Arc> top_order_visitor(&order, &acyclic);
    DfsVisit(*fst, &top_order_visitor, EpsilonArcFilter<Arc>());
    // Sanity check: should be acyclic if property bit is set.
    if (!acyclic) {
      FSTERROR() << "RmEpsilon: Inconsistent acyclic property bit";
      fst->SetProperties(kError, kError);
      return;
    }
    states.resize(order.size());
    for (StateId i = 0; i < order.size(); i++) states[order[i]] = i;
  } else {
    uint64 props;
    std::vector<StateId> scc;
    SccVisitor<Arc> scc_visitor(&scc, nullptr, nullptr, &props);
    DfsVisit(*fst, &scc_visitor, EpsilonArcFilter<Arc>());
    std::vector<StateId> first(scc.size(), kNoStateId);
    std::vector<StateId> next(scc.size(), kNoStateId);
    for (StateId i = 0; i < scc.size(); i++) {
      if (first[scc[i]] != kNoStateId) next[i] = first[scc[i]];
      first[scc[i]] = i;
    }
    for (StateId i = 0; i < first.size(); i++) {
      for (auto j = first[i]; j != kNoStateId; j = next[j]) {
        states.push_back(j);
      }
    }
  }
  internal::RmEpsilonState<Arc, Queue> rmeps_state(*fst, distance, opts);
  while (!states.empty()) {
    const auto state = states.back();
    states.pop_back();
    if (!noneps_in[state] &&
        (opts.connect || opts.weight_threshold != Weight::Zero() ||
         opts.state_threshold != kNoStateId)) {
      continue;
    }
    rmeps_state.Expand(state);
    fst->SetFinal(state, rmeps_state.Final());
    fst->DeleteArcs(state);
    auto &arcs = rmeps_state.Arcs();
    fst->ReserveArcs(state, arcs.size());
    while (!arcs.empty()) {
      fst->AddArc(state, arcs.back());
      arcs.pop_back();
    }
  }
  if (opts.connect || opts.weight_threshold != Weight::Zero() ||
      opts.state_threshold != kNoStateId) {
    for (size_t s = 0; s < fst->NumStates(); ++s) {
      if (!noneps_in[s]) fst->DeleteArcs(s);
    }
  }
  if (rmeps_state.Error()) fst->SetProperties(kError, kError);
  fst->SetProperties(
      RmEpsilonProperties(fst->Properties(kFstProperties, false)),
      kFstProperties);
  if (opts.weight_threshold != Weight::Zero() ||
      opts.state_threshold != kNoStateId) {
    Prune(fst, opts.weight_threshold, opts.state_threshold);
  }
  if (opts.connect && opts.weight_threshold == Weight::Zero() &&
      opts.state_threshold == kNoStateId) {
    Connect(fst);
  }
}

// Removes epsilon-transitions (when both the input and output label
// are an epsilon) from a transducer. The result will be an equivalent
// FST that has no such epsilon transitions. This version modifies its
// input. It has a simplified interface; see above for a version that
// allows finer control.
//
// Complexity:
//
// - Time:
//
//   Unweighted: O(v^2 + ve).
//   Acyclic: O(v^2 + V e).
//   Tropical semiring: O(v^2 log V + ve).
//   General: exponential.
//
// - Space: O(vE)
//
// where v is the number of states visited and e is the number of arcs visited.
//
// For more information, see:
//
// Mohri, M. 2002. Generic epsilon-removal and input epsilon-normalization
// algorithms for weighted transducers. International Journal of Computer
// Science 13(1): 129-143.
template <class Arc>
void RmEpsilon(MutableFst<Arc> *fst, bool connect = true,
               typename Arc::Weight weight_threshold = Arc::Weight::Zero(),
               typename Arc::StateId state_threshold = kNoStateId,
               float delta = kShortestDelta) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  std::vector<Weight> distance;
  AutoQueue<StateId> state_queue(*fst, &distance, EpsilonArcFilter<Arc>());
  RmEpsilonOptions<Arc, AutoQueue<StateId>> opts(
      &state_queue, delta, connect, weight_threshold, state_threshold);
  RmEpsilon(fst, &distance, opts);
}

struct RmEpsilonFstOptions : CacheOptions {
  float delta;

  explicit RmEpsilonFstOptions(const CacheOptions &opts,
                               float delta = kShortestDelta)
      : CacheOptions(opts), delta(delta) {}

  explicit RmEpsilonFstOptions(float delta = kShortestDelta) : delta(delta) {}
};

namespace internal {

// Implementation of delayed RmEpsilonFst.
template <class Arc>
class RmEpsilonFstImpl : public CacheImpl<Arc> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = DefaultCacheStore<Arc>;
  using State = typename Store::State;

  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  using CacheBaseImpl<CacheState<Arc>>::HasArcs;
  using CacheBaseImpl<CacheState<Arc>>::HasFinal;
  using CacheBaseImpl<CacheState<Arc>>::HasStart;
  using CacheBaseImpl<CacheState<Arc>>::PushArc;
  using CacheBaseImpl<CacheState<Arc>>::SetArcs;
  using CacheBaseImpl<CacheState<Arc>>::SetFinal;
  using CacheBaseImpl<CacheState<Arc>>::SetStart;

  RmEpsilonFstImpl(const Fst<Arc> &fst, const RmEpsilonFstOptions &opts)
      : CacheImpl<Arc>(opts),
        fst_(fst.Copy()),
        delta_(opts.delta),
        rmeps_state_(
            *fst_, &distance_,
            RmEpsilonOptions<Arc, FifoQueue<StateId>>(&queue_, delta_, false)) {
    SetType("rmepsilon");
    SetProperties(
        RmEpsilonProperties(fst.Properties(kFstProperties, false), true),
        kCopyProperties);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  RmEpsilonFstImpl(const RmEpsilonFstImpl &impl)
      : CacheImpl<Arc>(impl),
        fst_(impl.fst_->Copy(true)),
        delta_(impl.delta_),
        rmeps_state_(
            *fst_, &distance_,
            RmEpsilonOptions<Arc, FifoQueue<StateId>>(&queue_, delta_, false)) {
    SetType("rmepsilon");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  StateId Start() {
    if (!HasStart()) SetStart(fst_->Start());
    return CacheImpl<Arc>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) Expand(s);
    return CacheImpl<Arc>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<Arc>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<Arc>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<Arc>::NumOutputEpsilons(s);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) &&
        (fst_->Properties(kError, false) || rmeps_state_.Error())) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl<Arc>::InitArcIterator(s, data);
  }

  void Expand(StateId s) {
    rmeps_state_.Expand(s);
    SetFinal(s, rmeps_state_.Final());
    auto &arcs = rmeps_state_.Arcs();
    while (!arcs.empty()) {
      PushArc(s, arcs.back());
      arcs.pop_back();
    }
    SetArcs(s);
  }

 private:
  std::unique_ptr<const Fst<Arc>> fst_;
  float delta_;
  std::vector<Weight> distance_;
  FifoQueue<StateId> queue_;
  internal::RmEpsilonState<Arc, FifoQueue<StateId>> rmeps_state_;
};

}  // namespace internal

// Removes epsilon-transitions (when both the input and output label are an
// epsilon) from a transducer. The result will be an equivalent FST that has no
// such epsilon transitions. This version is a
// delayed FST.
//
// Complexity:
//
// - Time:
//   Unweighted: O(v^2 + ve).
//   General: exponential.
//
// - Space: O(vE)
//
// where v is the number of states visited and e is the number of arcs visited.
// Constant time to visit an input state or arc is assumed and exclusive of
// caching.
//
// For more information, see:
//
// Mohri, M. 2002. Generic epsilon-removal and input epsilon-normalization
// algorithms for weighted transducers. International Journal of Computer
// Science 13(1): 129-143.
//
// This class attaches interface to implementation and handles
// reference counting, delegating most methods to ImplToFst.
template <class A>
class RmEpsilonFst : public ImplToFst<internal::RmEpsilonFstImpl<A>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using Store = DefaultCacheStore<Arc>;
  using State = typename Store::State;
  using Impl = internal::RmEpsilonFstImpl<Arc>;

  friend class ArcIterator<RmEpsilonFst<Arc>>;
  friend class StateIterator<RmEpsilonFst<Arc>>;

  explicit RmEpsilonFst(const Fst<Arc> &fst)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, RmEpsilonFstOptions())) {}

  RmEpsilonFst(const Fst<A> &fst, const RmEpsilonFstOptions &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, opts)) {}

  // See Fst<>::Copy() for doc.
  RmEpsilonFst(const RmEpsilonFst<Arc> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this RmEpsilonFst. See Fst<>::Copy() for further doc.
  RmEpsilonFst<Arc> *Copy(bool safe = false) const override {
    return new RmEpsilonFst<Arc>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  RmEpsilonFst &operator=(const RmEpsilonFst &) = delete;
};

// Specialization for RmEpsilonFst.
template <class Arc>
class StateIterator<RmEpsilonFst<Arc>>
    : public CacheStateIterator<RmEpsilonFst<Arc>> {
 public:
  explicit StateIterator(const RmEpsilonFst<Arc> &fst)
      : CacheStateIterator<RmEpsilonFst<Arc>>(fst, fst.GetMutableImpl()) {}
};

// Specialization for RmEpsilonFst.
template <class Arc>
class ArcIterator<RmEpsilonFst<Arc>>
    : public CacheArcIterator<RmEpsilonFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const RmEpsilonFst<Arc> &fst, StateId s)
      : CacheArcIterator<RmEpsilonFst<Arc>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class Arc>
inline void RmEpsilonFst<Arc>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<RmEpsilonFst<Arc>>(*this);
}

// Useful alias when using StdArc.
using StdRmEpsilonFst = RmEpsilonFst<StdArc>;

}  // namespace fst

#endif  // FST_RMEPSILON_H_
