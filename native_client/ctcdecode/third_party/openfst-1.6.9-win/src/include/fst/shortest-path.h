// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions to find shortest paths in an FST.

#ifndef FST_SHORTEST_PATH_H_
#define FST_SHORTEST_PATH_H_

#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/determinize.h>
#include <fst/queue.h>
#include <fst/shortest-distance.h>
#include <fst/test-properties.h>


namespace fst {

template <class Arc, class Queue, class ArcFilter>
struct ShortestPathOptions
    : public ShortestDistanceOptions<Arc, Queue, ArcFilter> {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  int32_t nshortest;    // Returns n-shortest paths.
  bool unique;        // Only returns paths with distinct input strings.
  bool has_distance;  // Distance vector already contains the
                      // shortest distance from the initial state.
  bool first_path;    // Single shortest path stops after finding the first
                      // path to a final state; that path is the shortest path
                      // only when:
                      // (1) using the ShortestFirstQueue with all the weights
                      // in the FST being between One() and Zero() according to
                      // NaturalLess or when
                      // (2) using the NaturalAStarQueue with an admissible
                      // and consistent estimate.
  Weight weight_threshold;  // Pruning weight threshold.
  StateId state_threshold;  // Pruning state threshold.

  ShortestPathOptions(Queue *queue, ArcFilter filter, int32_t nshortest = 1,
                      bool unique = false, bool has_distance = false,
                      float delta = kShortestDelta, bool first_path = false,
                      Weight weight_threshold = Weight::Zero(),
                      StateId state_threshold = kNoStateId)
      : ShortestDistanceOptions<Arc, Queue, ArcFilter>(queue, filter,
                                                       kNoStateId, delta),
        nshortest(nshortest),
        unique(unique),
        has_distance(has_distance),
        first_path(first_path),
        weight_threshold(std::move(weight_threshold)),
        state_threshold(state_threshold) {}
};

namespace internal {

constexpr size_t kNoArc = -1;

// Helper function for SingleShortestPath building the shortest path as a left-
// to-right machine backwards from the best final state. It takes the input
// FST passed to SingleShortestPath and the parent vector and f_parent returned
// by that function, and builds the result into the provided output mutable FS
// This is not normally called by users; see ShortestPath instead.
template <class Arc>
void SingleShortestPathBacktrace(
    const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
    const std::vector<std::pair<typename Arc::StateId, size_t>> &parent,
    typename Arc::StateId f_parent) {
  using StateId = typename Arc::StateId;
  ofst->DeleteStates();
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  StateId s_p = kNoStateId;
  StateId d_p = kNoStateId;
  for (StateId state = f_parent, d = kNoStateId; state != kNoStateId;
       d = state, state = parent[state].first) {
    d_p = s_p;
    s_p = ofst->AddState();
    if (d == kNoStateId) {
      ofst->SetFinal(s_p, ifst.Final(f_parent));
    } else {
      ArcIterator<Fst<Arc>> aiter(ifst, state);
      aiter.Seek(parent[d].second);
      auto arc = aiter.Value();
      arc.nextstate = d_p;
      ofst->AddArc(s_p, arc);
    }
  }
  ofst->SetStart(s_p);
  if (ifst.Properties(kError, false)) ofst->SetProperties(kError, kError);
  ofst->SetProperties(
      ShortestPathProperties(ofst->Properties(kFstProperties, false), true),
      kFstProperties);
}

// Helper function for SingleShortestPath building a tree of shortest paths to
// every final state in the input FST. It takes the input FST and parent values
// computed by SingleShortestPath and builds into the output mutable FST the
// subtree of ifst that consists only of the best paths to all final states.
// This is not normally called by users; see ShortestPath instead.
template <class Arc>
void SingleShortestTree(
    const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
    const std::vector<std::pair<typename Arc::StateId, size_t>> &parent) {
  ofst->DeleteStates();
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  ofst->SetStart(ifst.Start());
  for (StateIterator<Fst<Arc>> siter(ifst); !siter.Done(); siter.Next()) {
    ofst->AddState();
    ofst->SetFinal(siter.Value(), ifst.Final(siter.Value()));
  }
  for (const auto &pair : parent) {
    if (pair.first != kNoStateId && pair.second != kNoArc) {
      ArcIterator<Fst<Arc>> aiter(ifst, pair.first);
      aiter.Seek(pair.second);
      ofst->AddArc(pair.first, aiter.Value());
    }
  }
  if (ifst.Properties(kError, false)) ofst->SetProperties(kError, kError);
  ofst->SetProperties(
      ShortestPathProperties(ofst->Properties(kFstProperties, false), true),
      kFstProperties);
}

// Implements the stopping criterion when ShortestPathOptions::first_path
// is set to true:
//   operator()(s, d, f) == true
//   iff every successful path through state 's' has a cost greater or equal
//   to 'f' under the assumption that 'd' is the shortest distance to state 's'.
// Correct when using the ShortestFirstQueue with all the weights in the FST
// being between One() and Zero() according to NaturalLess
template <typename S, typename W, typename Queue>
struct FirstPathSelect {
  FirstPathSelect(const Queue &) {}
  bool operator()(S s, W d, W f) const { return f == Plus(d, f); }
};

// Specialisation for A*.
// Correct when the estimate is admissible and consistent.
template <typename S, typename W, typename Estimate>
class FirstPathSelect<S, W, NaturalAStarQueue<S, W, Estimate>> {
 public:
  using Queue = NaturalAStarQueue<S, W, Estimate>;

  FirstPathSelect(const Queue &state_queue)
    : estimate_(state_queue.GetCompare().GetEstimate()) {}

  bool operator()(S s, W d, W f) const {
    return f == Plus(Times(d, estimate_(s)), f);
  }

 private:
  const Estimate &estimate_;
};

// Shortest-path algorithm. It builds the output mutable FST so that it contains
// the shortest path in the input FST; distance returns the shortest distances
// from the source state to each state in the input FST, and the options struct
// is
// used to specify options such as the queue discipline, the arc filter and
// delta. The super_final option is an output parameter indicating the final
// state, and the parent argument is used for the storage of the backtrace path
// for each state 1 to n, (i.e., the best previous state and the arc that
// transition to state n.) The shortest path is the lowest weight path w.r.t.
// the natural semiring order. The weights need to be right distributive and
// have the path (kPath) property. False is returned if an error is encountered.
//
// This is not normally called by users; see ShortestPath instead (with n = 1).
template <class Arc, class Queue, class ArcFilter>
bool SingleShortestPath(
    const Fst<Arc> &ifst, std::vector<typename Arc::Weight> *distance,
    const ShortestPathOptions<Arc, Queue, ArcFilter> &opts,
    typename Arc::StateId *f_parent,
    std::vector<std::pair<typename Arc::StateId, size_t>> *parent) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  static_assert(IsPath<Weight>::value, "Weight must have path property.");
  static_assert((Weight::Properties() & kRightSemiring) == kRightSemiring,
                "Weight must be right distributive.");
  parent->clear();
  *f_parent = kNoStateId;
  if (ifst.Start() == kNoStateId) return true;
  std::vector<bool> enqueued;
  auto state_queue = opts.state_queue;
  const auto source = (opts.source == kNoStateId) ? ifst.Start() : opts.source;
  bool final_seen = false;
  auto f_distance = Weight::Zero();
  distance->clear();
  state_queue->Clear();
  while (distance->size() < source) {
    distance->push_back(Weight::Zero());
    enqueued.push_back(false);
    parent->push_back(std::make_pair(kNoStateId, kNoArc));
  }
  distance->push_back(Weight::One());
  parent->push_back(std::make_pair(kNoStateId, kNoArc));
  state_queue->Enqueue(source);
  enqueued.push_back(true);
  while (!state_queue->Empty()) {
    const auto s = state_queue->Head();
    state_queue->Dequeue();
    enqueued[s] = false;
    const auto sd = (*distance)[s];
    // If we are using a shortest queue, no other path is going to be shorter
    // than f_distance at this point.
    using FirstPath = FirstPathSelect<StateId, Weight, Queue>;
    if (opts.first_path && final_seen &&
        FirstPath(*state_queue)(s, sd, f_distance)) {
      break;
    }
    if (ifst.Final(s) != Weight::Zero()) {
      const auto plus = Plus(f_distance, Times(sd, ifst.Final(s)));
      if (f_distance != plus) {
        f_distance = plus;
        *f_parent = s;
      }
      if (!f_distance.Member()) return false;
      final_seen = true;
    }
    for (ArcIterator<Fst<Arc>> aiter(ifst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      while (distance->size() <= arc.nextstate) {
        distance->push_back(Weight::Zero());
        enqueued.push_back(false);
        parent->push_back(std::make_pair(kNoStateId, kNoArc));
      }
      auto &nd = (*distance)[arc.nextstate];
      const auto weight = Times(sd, arc.weight);
      if (nd != Plus(nd, weight)) {
        nd = Plus(nd, weight);
        if (!nd.Member()) return false;
        (*parent)[arc.nextstate] = std::make_pair(s, aiter.Position());
        if (!enqueued[arc.nextstate]) {
          state_queue->Enqueue(arc.nextstate);
          enqueued[arc.nextstate] = true;
        } else {
          state_queue->Update(arc.nextstate);
        }
      }
    }
  }
  return true;
}

template <class StateId, class Weight>
class ShortestPathCompare {
 public:
  ShortestPathCompare(const std::vector<std::pair<StateId, Weight>> &pairs,
                      const std::vector<Weight> &distance, StateId superfinal,
                      float delta)
      : pairs_(pairs),
        distance_(distance),
        superfinal_(superfinal),
        delta_(delta) {}

  bool operator()(const StateId x, const StateId y) const {
    const auto &px = pairs_[x];
    const auto &py = pairs_[y];
    const auto wx = Times(PWeight(px.first), px.second);
    const auto wy = Times(PWeight(py.first), py.second);
    // Penalize complete paths to ensure correct results with inexact weights.
    // This forms a strict weak order so long as ApproxEqual(a, b) =>
    // ApproxEqual(a, c) for all c s.t. less_(a, c) && less_(c, b).
    if (px.first == superfinal_ && py.first != superfinal_) {
      return less_(wy, wx) || ApproxEqual(wx, wy, delta_);
    } else if (py.first == superfinal_ && px.first != superfinal_) {
      return less_(wy, wx) && !ApproxEqual(wx, wy, delta_);
    } else {
      return less_(wy, wx);
    }
  }

 private:
  Weight PWeight(StateId state) const {
    return (state == superfinal_)
               ? Weight::One()
               : (state < distance_.size()) ? distance_[state] : Weight::Zero();
  }

  const std::vector<std::pair<StateId, Weight>> &pairs_;
  const std::vector<Weight> &distance_;
  const StateId superfinal_;
  const float delta_;
  NaturalLess<Weight> less_;
};

// N-Shortest-path algorithm: implements the core n-shortest path algorithm.
// The output is built reversed. See below for versions with more options and
// *not reversed*.
//
// The output mutable FST contains the REVERSE of n'shortest paths in the input
// FST; distance must contain the shortest distance from each state to a final
// state in the input FST; delta is the convergence delta.
//
// The n-shortest paths are the n-lowest weight paths w.r.t. the natural
// semiring order. The single path that can be read from the ith of at most n
// transitions leaving the initial state of the input FST is the ith shortest
// path. Disregarding the initial state and initial transitions, the
// n-shortest paths, in fact, form a tree rooted at the single final state.
//
// The weights need to be left and right distributive (kSemiring) and have the
// path (kPath) property.
//
// Arc weights must satisfy the property that the sum of the weights of one or
// more paths from some state S to T is never Zero(). In particular, arc weights
// are never Zero().
//
// For more information, see:
//
// Mohri, M, and Riley, M. 2002. An efficient algorithm for the n-best-strings
// problem. In Proc. ICSLP.
//
// The algorithm relies on the shortest-distance algorithm. There are some
// issues with the pseudo-code as written in the paper (viz., line 11).
//
// IMPLEMENTATION NOTE: The input FST can be a delayed FST and at any state in
// its expansion the values of distance vector need only be defined at that time
// for the states that are known to exist.
template <class Arc, class RevArc>
void NShortestPath(const Fst<RevArc> &ifst, MutableFst<Arc> *ofst,
                   const std::vector<typename Arc::Weight> &distance,
                   int32_t nshortest, float delta = kShortestDelta,
                   typename Arc::Weight weight_threshold = Arc::Weight::Zero(),
                   typename Arc::StateId state_threshold = kNoStateId) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using Pair = std::pair<StateId, Weight>;
  static_assert((Weight::Properties() & kPath) == kPath,
                "Weight must have path property.");
  static_assert((Weight::Properties() & kSemiring) == kSemiring,
                "Weight must be distributive.");
  if (nshortest <= 0) return;
  ofst->DeleteStates();
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  // Each state in ofst corresponds to a path with weight w from the initial
  // state of ifst to a state s in ifst, that can be characterized by a pair
  // (s, w). The vector pairs maps each state in ofst to the corresponding
  // pair maps states in ofst to the corresponding pair (s, w).
  std::vector<Pair> pairs;
  // The supefinal state is denoted by kNoStateId. The distance from the
  // superfinal state to the final state is semiring One, so
  // `distance[kNoStateId]` is not needed.
  const ShortestPathCompare<StateId, Weight> compare(pairs, distance,
                                                     kNoStateId, delta);
  const NaturalLess<Weight> less;
  if (ifst.Start() == kNoStateId || distance.size() <= ifst.Start() ||
      distance[ifst.Start()] == Weight::Zero() ||
      less(weight_threshold, Weight::One()) || state_threshold == 0) {
    if (ifst.Properties(kError, false)) ofst->SetProperties(kError, kError);
    return;
  }
  ofst->SetStart(ofst->AddState());
  const auto final_state = ofst->AddState();
  ofst->SetFinal(final_state, Weight::One());
  while (pairs.size() <= final_state) {
    pairs.push_back(std::make_pair(kNoStateId, Weight::Zero()));
  }
  pairs[final_state] = std::make_pair(ifst.Start(), Weight::One());
  std::vector<StateId> heap;
  heap.push_back(final_state);
  const auto limit = Times(distance[ifst.Start()], weight_threshold);
  // r[s + 1], s state in fst, is the number of states in ofst which
  // corresponding pair contains s, i.e., it is number of paths computed so far
  // to s. Valid for s == kNoStateId (the superfinal state).
  std::vector<int> r;
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), compare);
    const auto state = heap.back();
    const auto p = pairs[state];
    heap.pop_back();
    const auto d =
        (p.first == kNoStateId)
            ? Weight::One()
            : (p.first < distance.size()) ? distance[p.first] : Weight::Zero();
    if (less(limit, Times(d, p.second)) ||
        (state_threshold != kNoStateId &&
         ofst->NumStates() >= state_threshold)) {
      continue;
    }
    while (r.size() <= p.first + 1) r.push_back(0);
    ++r[p.first + 1];
    if (p.first == kNoStateId) {
      ofst->AddArc(ofst->Start(), Arc(0, 0, Weight::One(), state));
    }
    if ((p.first == kNoStateId) && (r[p.first + 1] == nshortest)) break;
    if (r[p.first + 1] > nshortest) continue;
    if (p.first == kNoStateId) continue;
    for (ArcIterator<Fst<RevArc>> aiter(ifst, p.first); !aiter.Done();
         aiter.Next()) {
      const auto &rarc = aiter.Value();
      Arc arc(rarc.ilabel, rarc.olabel, rarc.weight.Reverse(), rarc.nextstate);
      const auto weight = Times(p.second, arc.weight);
      const auto next = ofst->AddState();
      pairs.push_back(std::make_pair(arc.nextstate, weight));
      arc.nextstate = state;
      ofst->AddArc(next, arc);
      heap.push_back(next);
      std::push_heap(heap.begin(), heap.end(), compare);
    }
    const auto final_weight = ifst.Final(p.first).Reverse();
    if (final_weight != Weight::Zero()) {
      const auto weight = Times(p.second, final_weight);
      const auto next = ofst->AddState();
      pairs.push_back(std::make_pair(kNoStateId, weight));
      ofst->AddArc(next, Arc(0, 0, final_weight, state));
      heap.push_back(next);
      std::push_heap(heap.begin(), heap.end(), compare);
    }
  }
  Connect(ofst);
  if (ifst.Properties(kError, false)) ofst->SetProperties(kError, kError);
  ofst->SetProperties(
      ShortestPathProperties(ofst->Properties(kFstProperties, false)),
      kFstProperties);
}

}  // namespace internal

// N-Shortest-path algorithm: this version allows finer control via the options
// argument. See below for a simpler interface. The output mutable FST contains
// the n-shortest paths in the input FST; the distance argument is used to
// return the shortest distances from the source state to each state in the
// input FST, and the options struct is used to specify the number of paths to
// return, whether they need to have distinct input strings, the queue
// discipline, the arc filter and the convergence delta.
//
// The n-shortest paths are the n-lowest weight paths w.r.t. the natural
// semiring order. The single path that can be read from the ith of at most n
// transitions leaving the initial state of the output FST is the ith shortest
// path.
// Disregarding the initial state and initial transitions, The n-shortest paths,
// in fact, form a tree rooted at the single final state.
//
// The weights need to be right distributive and have the path (kPath) property.
// They need to be left distributive as well for nshortest > 1.
//
// For more information, see:
//
// Mohri, M, and Riley, M. 2002. An efficient algorithm for the n-best-strings
// problem. In Proc. ICSLP.
//
// The algorithm relies on the shortest-distance algorithm. There are some
// issues with the pseudo-code as written in the paper (viz., line 11).
template <class Arc, class Queue, class ArcFilter,
          typename std::enable_if<IsPath<typename Arc::Weight>::value>::type * =
              nullptr>
void ShortestPath(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                  std::vector<typename Arc::Weight> *distance,
                  const ShortestPathOptions<Arc, Queue, ArcFilter> &opts) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using RevArc = ReverseArc<Arc>;
  if (opts.nshortest == 1) {
    std::vector<std::pair<StateId, size_t>> parent;
    StateId f_parent;
    if (internal::SingleShortestPath(ifst, distance, opts, &f_parent,
                                     &parent)) {
      internal::SingleShortestPathBacktrace(ifst, ofst, parent, f_parent);
    } else {
      ofst->SetProperties(kError, kError);
    }
    return;
  }
  if (opts.nshortest <= 0) return;
  if (!opts.has_distance) {
    ShortestDistance(ifst, distance, opts);
    if (distance->size() == 1 && !(*distance)[0].Member()) {
      ofst->SetProperties(kError, kError);
      return;
    }
  }
  // Algorithm works on the reverse of 'fst'; 'distance' is the distance to the
  // final state in 'rfst', 'ofst' is built as the reverse of the tree of
  // n-shortest path in 'rfst'.
  VectorFst<RevArc> rfst;
  Reverse(ifst, &rfst);
  auto d = Weight::Zero();
  for (ArcIterator<VectorFst<RevArc>> aiter(rfst, 0); !aiter.Done();
       aiter.Next()) {
    const auto &arc = aiter.Value();
    const auto state = arc.nextstate - 1;
    if (state < distance->size()) {
      d = Plus(d, Times(arc.weight.Reverse(), (*distance)[state]));
    }
  }
  // TODO(kbg): Avoid this expensive vector operation.
  distance->insert(distance->begin(), d);
  if (!opts.unique) {
    internal::NShortestPath(rfst, ofst, *distance, opts.nshortest, opts.delta,
                            opts.weight_threshold, opts.state_threshold);
  } else {
    std::vector<Weight> ddistance;
    DeterminizeFstOptions<RevArc> dopts(opts.delta);
    DeterminizeFst<RevArc> dfst(rfst, distance, &ddistance, dopts);
    internal::NShortestPath(dfst, ofst, ddistance, opts.nshortest, opts.delta,
                            opts.weight_threshold, opts.state_threshold);
  }
  // TODO(kbg): Avoid this expensive vector operation.
  distance->erase(distance->begin());
}

template <class Arc, class Queue, class ArcFilter,
          typename std::enable_if<!IsPath<typename Arc::Weight>::value>::type
              * = nullptr>
void ShortestPath(const Fst<Arc> &, MutableFst<Arc> *ofst,
                  std::vector<typename Arc::Weight> *,
                  const ShortestPathOptions<Arc, Queue, ArcFilter> &) {
  FSTERROR() << "ShortestPath: Weight needs to have the "
             << "path property and be distributive: " << Arc::Weight::Type();
  ofst->SetProperties(kError, kError);
}

// Shortest-path algorithm: simplified interface. See above for a version that
// allows finer control. The output mutable FST contains the n-shortest paths
// in the input FST. The queue discipline is automatically selected. When unique
// is true, only paths with distinct input label sequences are returned.
//
// The n-shortest paths are the n-lowest weight paths w.r.t. the natural
// semiring order. The single path that can be read from the ith of at most n
// transitions leaving the initial state of the output FST is the ith best path.
// The weights need to be right distributive and have the path (kPath) property.
template <class Arc>
void ShortestPath(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                  int32_t nshortest = 1, bool unique = false,
                  bool first_path = false,
                  typename Arc::Weight weight_threshold = Arc::Weight::Zero(),
                  typename Arc::StateId state_threshold = kNoStateId,
                  float delta = kShortestDelta) {
  using StateId = typename Arc::StateId;
  std::vector<typename Arc::Weight> distance;
  AnyArcFilter<Arc> arc_filter;
  AutoQueue<StateId> state_queue(ifst, &distance, arc_filter);
  const ShortestPathOptions<Arc, AutoQueue<StateId>, AnyArcFilter<Arc>> opts(
      &state_queue, arc_filter, nshortest, unique, false, delta, first_path,
      weight_threshold, state_threshold);
  ShortestPath(ifst, ofst, &distance, opts);
}

}  // namespace fst

#endif  // FST_SHORTEST_PATH_H_
