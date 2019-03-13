// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions implementing pruning.

#ifndef FST_PRUNE_H_
#define FST_PRUNE_H_

#include <type_traits>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/arcfilter.h>
#include <fst/heap.h>
#include <fst/shortest-distance.h>


namespace fst {
namespace internal {

template <class StateId, class Weight>
class PruneCompare {
 public:
  PruneCompare(const std::vector<Weight> &idistance,
               const std::vector<Weight> &fdistance)
      : idistance_(idistance), fdistance_(fdistance) {}

  bool operator()(const StateId x, const StateId y) const {
    const auto wx = Times(IDistance(x), FDistance(x));
    const auto wy = Times(IDistance(y), FDistance(y));
    return less_(wx, wy);
  }

 private:
  Weight IDistance(const StateId s) const {
    return s < idistance_.size() ? idistance_[s] : Weight::Zero();
  }

  Weight FDistance(const StateId s) const {
    return s < fdistance_.size() ? fdistance_[s] : Weight::Zero();
  }

  const std::vector<Weight> &idistance_;
  const std::vector<Weight> &fdistance_;
  NaturalLess<Weight> less_;
};

}  // namespace internal

template <class Arc, class ArcFilter>
struct PruneOptions {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit PruneOptions(const Weight &weight_threshold = Weight::Zero(),
                        StateId state_threshold = kNoStateId,
                        ArcFilter filter = ArcFilter(),
                        std::vector<Weight> *distance = nullptr,
                        float delta = kDelta, bool threshold_initial = false)
      : weight_threshold(std::move(weight_threshold)),
        state_threshold(state_threshold),
        filter(std::move(filter)),
        distance(distance),
        delta(delta),
        threshold_initial(threshold_initial) {}

  // Pruning weight threshold.
  Weight weight_threshold;
  // Pruning state threshold.
  StateId state_threshold;
  // Arc filter.
  ArcFilter filter;
  // If non-zero, passes in pre-computed shortest distance to final states.
  const std::vector<Weight> *distance;
  // Determines the degree of convergence required when computing shortest
  // distances.
  float delta;
  // Determines if the shortest path weight is left (true) or right
  // (false) multiplied by the threshold to get the limit for
  // keeping a state or arc (matters if the semiring is not
  // commutative).
  bool threshold_initial;
};

// Pruning algorithm: this version modifies its input and it takes an options
// class as an argument. After pruning the FST contains states and arcs that
// belong to a successful path in the FST whose weight is no more than the
// weight of the shortest path Times() the provided weight threshold. When the
// state threshold is not kNoStateId, the output FST is further restricted to
// have no more than the number of states in opts.state_threshold. Weights must
// have the path property. The weight of any cycle needs to be bounded; i.e.,
//
//   Plus(weight, Weight::One()) == Weight::One()
template <class Arc, class ArcFilter,
          typename std::enable_if<IsPath<typename Arc::Weight>::value>::type * =
              nullptr>
void Prune(MutableFst<Arc> *fst, const PruneOptions<Arc, ArcFilter> &opts =
                                     PruneOptions<Arc, ArcFilter>()) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using StateHeap = Heap<StateId, internal::PruneCompare<StateId, Weight>>;
  auto ns = fst->NumStates();
  if (ns < 1) return;
  std::vector<Weight> idistance(ns, Weight::Zero());
  std::vector<Weight> tmp;
  if (!opts.distance) {
    tmp.reserve(ns);
    ShortestDistance(*fst, &tmp, true, opts.delta);
  }
  const auto *fdistance = opts.distance ? opts.distance : &tmp;
  if ((opts.state_threshold == 0) || (fdistance->size() <= fst->Start()) ||
      ((*fdistance)[fst->Start()] == Weight::Zero())) {
    fst->DeleteStates();
    return;
  }
  internal::PruneCompare<StateId, Weight> compare(idistance, *fdistance);
  StateHeap heap(compare);
  std::vector<bool> visited(ns, false);
  std::vector<size_t> enqueued(ns, StateHeap::kNoKey);
  std::vector<StateId> dead;
  dead.push_back(fst->AddState());
  NaturalLess<Weight> less;
  auto s = fst->Start();
  const auto limit = opts.threshold_initial ?
      Times(opts.weight_threshold, (*fdistance)[s]) :
      Times((*fdistance)[s], opts.weight_threshold);
  StateId num_visited = 0;

  if (!less(limit, (*fdistance)[s])) {
    idistance[s] = Weight::One();
    enqueued[s] = heap.Insert(s);
    ++num_visited;
  }
  while (!heap.Empty()) {
    s = heap.Top();
    heap.Pop();
    enqueued[s] = StateHeap::kNoKey;
    visited[s] = true;
    if (less(limit, Times(idistance[s], fst->Final(s)))) {
      fst->SetFinal(s, Weight::Zero());
    }
    for (MutableArcIterator<MutableFst<Arc>> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      auto arc = aiter.Value();  // Copy intended.
      if (!opts.filter(arc)) continue;
      const auto weight = Times(Times(idistance[s], arc.weight),
                                arc.nextstate < fdistance->size() ?
                                (*fdistance)[arc.nextstate] : Weight::Zero());
      if (less(limit, weight)) {
        arc.nextstate = dead[0];
        aiter.SetValue(arc);
        continue;
      }
      if (less(Times(idistance[s], arc.weight), idistance[arc.nextstate])) {
        idistance[arc.nextstate] = Times(idistance[s], arc.weight);
      }
      if (visited[arc.nextstate]) continue;
      if ((opts.state_threshold != kNoStateId) &&
          (num_visited >= opts.state_threshold)) {
        continue;
      }
      if (enqueued[arc.nextstate] == StateHeap::kNoKey) {
        enqueued[arc.nextstate] = heap.Insert(arc.nextstate);
        ++num_visited;
      } else {
        heap.Update(enqueued[arc.nextstate], arc.nextstate);
      }
    }
  }
  for (StateId i = 0; i < visited.size(); ++i) {
    if (!visited[i]) dead.push_back(i);
  }
  fst->DeleteStates(dead);
}

template <class Arc, class ArcFilter,
          typename std::enable_if<!IsPath<typename Arc::Weight>::value>::type
              * = nullptr>
void Prune(MutableFst<Arc> *fst, const PruneOptions<Arc, ArcFilter> &opts =
                                     PruneOptions<Arc, ArcFilter>()) {
  FSTERROR() << "Prune: Weight needs to have the path property: "
             << Arc::Weight::Type();
  fst->SetProperties(kError, kError);
}

// Pruning algorithm: this version modifies its input and takes the
// pruning threshold as an argument. It deletes states and arcs in the
// FST that do not belong to a successful path whose weight is more
// than the weight of the shortest path Times() the provided weight
// threshold. When the state threshold is not kNoStateId, the output
// FST is further restricted to have no more than the number of states
// in opts.state_threshold. Weights must have the path property. The
// weight of any cycle needs to be bounded; i.e.,
//
//   Plus(weight, Weight::One()) == Weight::One()
template <class Arc>
void Prune(MutableFst<Arc> *fst, typename Arc::Weight weight_threshold,
           typename Arc::StateId state_threshold = kNoStateId,
           float delta = kDelta) {
  const PruneOptions<Arc, AnyArcFilter<Arc>> opts(
      weight_threshold, state_threshold, AnyArcFilter<Arc>(), nullptr, delta);
  Prune(fst, opts);
}

// Pruning algorithm: this version writes the pruned input FST to an
// output MutableFst and it takes an options class as an argument. The
// output FST contains states and arcs that belong to a successful
// path in the input FST whose weight is more than the weight of the
// shortest path Times() the provided weight threshold. When the state
// threshold is not kNoStateId, the output FST is further restricted
// to have no more than the number of states in
// opts.state_threshold. Weights have the path property.  The weight
// of any cycle needs to be bounded; i.e.,
//
//   Plus(weight, Weight::One()) == Weight::One()
template <class Arc, class ArcFilter,
          typename std::enable_if<IsPath<typename Arc::Weight>::value>::type * =
              nullptr>
void Prune(
    const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
    const PruneOptions<Arc, ArcFilter> &opts = PruneOptions<Arc, ArcFilter>()) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using StateHeap = Heap<StateId, internal::PruneCompare<StateId, Weight>>;
  ofst->DeleteStates();
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  if (ifst.Start() == kNoStateId) return;
  NaturalLess<Weight> less;
  if (less(opts.weight_threshold, Weight::One()) ||
      (opts.state_threshold == 0)) {
    return;
  }
  std::vector<Weight> idistance;
  std::vector<Weight> tmp;
  if (!opts.distance) ShortestDistance(ifst, &tmp, true, opts.delta);
  const auto *fdistance = opts.distance ? opts.distance : &tmp;
  if ((fdistance->size() <= ifst.Start()) ||
      ((*fdistance)[ifst.Start()] == Weight::Zero())) {
    return;
  }
  internal::PruneCompare<StateId, Weight> compare(idistance, *fdistance);
  StateHeap heap(compare);
  std::vector<StateId> copy;
  std::vector<size_t> enqueued;
  std::vector<bool> visited;
  auto s = ifst.Start();
  const auto limit = opts.threshold_initial ?
      Times(opts.weight_threshold, (*fdistance)[s]) :
      Times((*fdistance)[s], opts.weight_threshold);
  while (copy.size() <= s) copy.push_back(kNoStateId);
  copy[s] = ofst->AddState();
  ofst->SetStart(copy[s]);
  while (idistance.size() <= s) idistance.push_back(Weight::Zero());
  idistance[s] = Weight::One();
  while (enqueued.size() <= s) {
    enqueued.push_back(StateHeap::kNoKey);
    visited.push_back(false);
  }
  enqueued[s] = heap.Insert(s);
  while (!heap.Empty()) {
    s = heap.Top();
    heap.Pop();
    enqueued[s] = StateHeap::kNoKey;
    visited[s] = true;
    if (!less(limit, Times(idistance[s], ifst.Final(s)))) {
      ofst->SetFinal(copy[s], ifst.Final(s));
    }
    for (ArcIterator<Fst<Arc>> aiter(ifst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      if (!opts.filter(arc)) continue;
      const auto weight = Times(Times(idistance[s], arc.weight),
                                arc.nextstate < fdistance->size() ?
                                (*fdistance)[arc.nextstate] : Weight::Zero());
      if (less(limit, weight)) continue;
      if ((opts.state_threshold != kNoStateId) &&
          (ofst->NumStates() >= opts.state_threshold)) {
        continue;
      }
      while (idistance.size() <= arc.nextstate) {
        idistance.push_back(Weight::Zero());
      }
      if (less(Times(idistance[s], arc.weight), idistance[arc.nextstate])) {
        idistance[arc.nextstate] = Times(idistance[s], arc.weight);
      }
      while (copy.size() <= arc.nextstate) copy.push_back(kNoStateId);
      if (copy[arc.nextstate] == kNoStateId) {
        copy[arc.nextstate] = ofst->AddState();
      }
      ofst->AddArc(copy[s], Arc(arc.ilabel, arc.olabel, arc.weight,
                                copy[arc.nextstate]));
      while (enqueued.size() <= arc.nextstate) {
        enqueued.push_back(StateHeap::kNoKey);
        visited.push_back(false);
      }
      if (visited[arc.nextstate]) continue;
      if (enqueued[arc.nextstate] == StateHeap::kNoKey) {
        enqueued[arc.nextstate] = heap.Insert(arc.nextstate);
      } else {
        heap.Update(enqueued[arc.nextstate], arc.nextstate);
      }
    }
  }
}

template <class Arc, class ArcFilter,
          typename std::enable_if<!IsPath<typename Arc::Weight>::value>::type
              * = nullptr>
void Prune(const Fst<Arc> &, MutableFst<Arc> *ofst,
           const PruneOptions<Arc, ArcFilter> &) {
  FSTERROR() << "Prune: Weight needs to have the path property: "
             << Arc::Weight::Type();
  ofst->SetProperties(kError, kError);
}

// Pruning algorithm: this version writes the pruned input FST to an
// output MutableFst and simply takes the pruning threshold as an
// argument. The output FST contains states and arcs that belong to a
// successful path in the input FST whose weight is no more than the
// weight of the shortest path Times() the provided weight
// threshold. When the state threshold is not kNoStateId, the output
// FST is further restricted to have no more than the number of states
// in opts.state_threshold. Weights must have the path property. The
// weight of any cycle needs to be bounded; i.e.,
//
// Plus(weight, Weight::One()) = Weight::One();
template <class Arc>
void Prune(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
           typename Arc::Weight weight_threshold,
           typename Arc::StateId state_threshold = kNoStateId,
           float delta = kDelta) {
  const PruneOptions<Arc, AnyArcFilter<Arc>> opts(
      weight_threshold, state_threshold, AnyArcFilter<Arc>(), nullptr, delta);
  Prune(ifst, ofst, opts);
}

}  // namespace fst

#endif  // FST_PRUNE_H_
