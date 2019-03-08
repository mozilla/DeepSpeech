// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to find shortest distance in an FST.

#ifndef FST_SHORTEST_DISTANCE_H_
#define FST_SHORTEST_DISTANCE_H_

#include <deque>
#include <vector>

#include <fst/log.h>

#include <fst/arcfilter.h>
#include <fst/cache.h>
#include <fst/queue.h>
#include <fst/reverse.h>
#include <fst/test-properties.h>


namespace fst {

// A representable float for shortest distance and shortest path algorithms.
constexpr float kShortestDelta = 1e-6;

template <class Arc, class Queue, class ArcFilter>
struct ShortestDistanceOptions {
  using StateId = typename Arc::StateId;

  Queue *state_queue;    // Queue discipline used; owned by caller.
  ArcFilter arc_filter;  // Arc filter (e.g., limit to only epsilon graph).
  StateId source;        // If kNoStateId, use the FST's initial state.
  float delta;           // Determines the degree of convergence required
  bool first_path;       // For a semiring with the path property (o.w.
                         // undefined), compute the shortest-distances along
                         // along the first path to a final state found
                         // by the algorithm. That path is the shortest-path
                         // only if the FST has a unique final state (or all
                         // the final states have the same final weight), the
                         // queue discipline is shortest-first and all the
                         // weights in the FST are between One() and Zero()
                         // according to NaturalLess.

  ShortestDistanceOptions(Queue *state_queue, ArcFilter arc_filter,
                          StateId source = kNoStateId,
                          float delta = kShortestDelta,
                          bool first_path = false)
      : state_queue(state_queue),
        arc_filter(arc_filter),
        source(source),
        delta(delta),
        first_path(first_path) {}
};

namespace internal {

// Computation state of the shortest-distance algorithm. Reusable information
// is maintained across calls to member function ShortestDistance(source) when
// retain is true for improved efficiency when calling multiple times from
// different source states (e.g., in epsilon removal). Contrary to the usual
// conventions, fst may not be freed before this class. Vector distance
// should not be modified by the user between these calls. The Error() method
// returns true iff an error was encountered.
template <class Arc, class Queue, class ArcFilter>
class ShortestDistanceState {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  ShortestDistanceState(
      const Fst<Arc> &fst, std::vector<Weight> *distance,
      const ShortestDistanceOptions<Arc, Queue, ArcFilter> &opts, bool retain)
      : fst_(fst),
        distance_(distance),
        state_queue_(opts.state_queue),
        arc_filter_(opts.arc_filter),
        delta_(opts.delta),
        first_path_(opts.first_path),
        retain_(retain),
        source_id_(0),
        error_(false) {
    distance_->clear();
  }

  void ShortestDistance(StateId source);

  bool Error() const { return error_; }

 private:
  const Fst<Arc> &fst_;
  std::vector<Weight> *distance_;
  Queue *state_queue_;
  ArcFilter arc_filter_;
  const float delta_;
  const bool first_path_;
  const bool retain_;  // Retain and reuse information across calls.

  std::vector<Adder<Weight>> adder_;   // Sums distance_ accurately.
  std::vector<Adder<Weight>> radder_;  // Relaxation distance.
  std::vector<bool> enqueued_;         // Is state enqueued?
  std::vector<StateId> sources_;       // Source ID for ith state in distance_,
                                       // (r)adder_, and enqueued_ if retained.
  StateId source_id_;                  // Unique ID characterizing each call.
  bool error_;
};

// Compute the shortest distance; if source is kNoStateId, uses the initial
// state of the FST.
template <class Arc, class Queue, class ArcFilter>
void ShortestDistanceState<Arc, Queue, ArcFilter>::ShortestDistance(
    StateId source) {
  if (fst_.Start() == kNoStateId) {
    if (fst_.Properties(kError, false)) error_ = true;
    return;
  }
  if (!(Weight::Properties() & kRightSemiring)) {
    FSTERROR() << "ShortestDistance: Weight needs to be right distributive: "
               << Weight::Type();
    error_ = true;
    return;
  }
  if (first_path_ && !(Weight::Properties() & kPath)) {
    FSTERROR() << "ShortestDistance: The first_path option is disallowed when "
               << "Weight does not have the path property: " << Weight::Type();
    error_ = true;
    return;
  }
  state_queue_->Clear();
  if (!retain_) {
    distance_->clear();
    adder_.clear();
    radder_.clear();
    enqueued_.clear();
  }
  if (source == kNoStateId) source = fst_.Start();
  while (distance_->size() <= source) {
    distance_->push_back(Weight::Zero());
    adder_.push_back(Adder<Weight>());
    radder_.push_back(Adder<Weight>());
    enqueued_.push_back(false);
  }
  if (retain_) {
    while (sources_.size() <= source) sources_.push_back(kNoStateId);
    sources_[source] = source_id_;
  }
  (*distance_)[source] = Weight::One();
  adder_[source].Reset(Weight::One());
  radder_[source].Reset(Weight::One());
  enqueued_[source] = true;
  state_queue_->Enqueue(source);
  while (!state_queue_->Empty()) {
    const auto state = state_queue_->Head();
    state_queue_->Dequeue();
    while (distance_->size() <= state) {
      distance_->push_back(Weight::Zero());
      adder_.push_back(Adder<Weight>());
      radder_.push_back(Adder<Weight>());
      enqueued_.push_back(false);
    }
    if (first_path_ && (fst_.Final(state) != Weight::Zero())) break;
    enqueued_[state] = false;
    const auto r = radder_[state].Sum();
    radder_[state].Reset();
    for (ArcIterator<Fst<Arc>> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
      if (!arc_filter_(arc)) continue;
      while (distance_->size() <= arc.nextstate) {
        distance_->push_back(Weight::Zero());
        adder_.push_back(Adder<Weight>());
        radder_.push_back(Adder<Weight>());
        enqueued_.push_back(false);
      }
      if (retain_) {
        while (sources_.size() <= arc.nextstate) sources_.push_back(kNoStateId);
        if (sources_[arc.nextstate] != source_id_) {
          (*distance_)[arc.nextstate] = Weight::Zero();
          adder_[arc.nextstate].Reset();
          radder_[arc.nextstate].Reset();
          enqueued_[arc.nextstate] = false;
          sources_[arc.nextstate] = source_id_;
        }
      }
      auto &nd = (*distance_)[arc.nextstate];
      auto &na = adder_[arc.nextstate];
      auto &nr = radder_[arc.nextstate];
      auto weight = Times(r, arc.weight);
      if (!ApproxEqual(nd, Plus(nd, weight), delta_)) {
        nd = na.Add(weight);
        nr.Add(weight);
        if (!nd.Member() || !nr.Sum().Member()) {
          error_ = true;
          return;
        }
        if (!enqueued_[arc.nextstate]) {
          state_queue_->Enqueue(arc.nextstate);
          enqueued_[arc.nextstate] = true;
        } else {
          state_queue_->Update(arc.nextstate);
        }
      }
    }
  }
  ++source_id_;
  if (fst_.Properties(kError, false)) error_ = true;
}

}  // namespace internal

// Shortest-distance algorithm: this version allows fine control
// via the options argument. See below for a simpler interface.
//
// This computes the shortest distance from the opts.source state to each
// visited state S and stores the value in the distance vector. An
// unvisited state S has distance Zero(), which will be stored in the
// distance vector if S is less than the maximum visited state. The state
// queue discipline, arc filter, and convergence delta are taken in the
// options argument. The distance vector will contain a unique element for
// which Member() is false if an error was encountered.
//
// The weights must must be right distributive and k-closed (i.e., 1 +
// x + x^2 + ... + x^(k +1) = 1 + x + x^2 + ... + x^k).
//
// Complexity:
//
// Depends on properties of the semiring and the queue discipline.
//
// For more information, see:
//
// Mohri, M. 2002. Semiring framework and algorithms for shortest-distance
// problems, Journal of Automata, Languages and
// Combinatorics 7(3): 321-350, 2002.
template <class Arc, class Queue, class ArcFilter>
void ShortestDistance(
    const Fst<Arc> &fst, std::vector<typename Arc::Weight> *distance,
    const ShortestDistanceOptions<Arc, Queue, ArcFilter> &opts) {
  internal::ShortestDistanceState<Arc, Queue, ArcFilter> sd_state(fst, distance,
                                                                  opts, false);
  sd_state.ShortestDistance(opts.source);
  if (sd_state.Error()) {
    distance->clear();
    distance->resize(1, Arc::Weight::NoWeight());
  }
}

// Shortest-distance algorithm: simplified interface. See above for a version
// that permits finer control.
//
// If reverse is false, this computes the shortest distance from the initial
// state to each state S and stores the value in the distance vector. If
// reverse is true, this computes the shortest distance from each state to the
// final states. An unvisited state S has distance Zero(), which will be stored
// in the distance vector if S is less than the maximum visited state. The
// state queue discipline is automatically-selected. The distance vector will
// contain a unique element for which Member() is false if an error was
// encountered.
//
// The weights must must be right (left) distributive if reverse is false (true)
// and k-closed (i.e., 1 + x + x^2 + ... + x^(k +1) = 1 + x + x^2 + ... + x^k).
//
// Arc weights must satisfy the property that the sum of the weights of one or
// more paths from some state S to T is never Zero(). In particular, arc weights
// are never Zero().
//
// Complexity:
//
// Depends on properties of the semiring and the queue discipline.
//
// For more information, see:
//
// Mohri, M. 2002. Semiring framework and algorithms for
// shortest-distance problems, Journal of Automata, Languages and
// Combinatorics 7(3): 321-350, 2002.
template <class Arc>
void ShortestDistance(const Fst<Arc> &fst,
                      std::vector<typename Arc::Weight> *distance,
                      bool reverse = false, float delta = kShortestDelta) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  if (!reverse) {
    AnyArcFilter<Arc> arc_filter;
    AutoQueue<StateId> state_queue(fst, distance, arc_filter);
    const ShortestDistanceOptions<Arc, AutoQueue<StateId>, AnyArcFilter<Arc>>
        opts(&state_queue, arc_filter, kNoStateId, delta);
    ShortestDistance(fst, distance, opts);
  } else {
    using ReverseArc = ReverseArc<Arc>;
    using ReverseWeight = typename ReverseArc::Weight;
    AnyArcFilter<ReverseArc> rarc_filter;
    VectorFst<ReverseArc> rfst;
    Reverse(fst, &rfst);
    std::vector<ReverseWeight> rdistance;
    AutoQueue<StateId> state_queue(rfst, &rdistance, rarc_filter);
    const ShortestDistanceOptions<ReverseArc, AutoQueue<StateId>,
                                  AnyArcFilter<ReverseArc>>
        ropts(&state_queue, rarc_filter, kNoStateId, delta);
    ShortestDistance(rfst, &rdistance, ropts);
    distance->clear();
    if (rdistance.size() == 1 && !rdistance[0].Member()) {
      distance->resize(1, Arc::Weight::NoWeight());
      return;
    }
    while (distance->size() < rdistance.size() - 1) {
      distance->push_back(rdistance[distance->size() + 1].Reverse());
    }
  }
}

// Return the sum of the weight of all successful paths in an FST, i.e., the
// shortest-distance from the initial state to the final states. Returns a
// weight such that Member() is false if an error was encountered.
template <class Arc>
typename Arc::Weight ShortestDistance(const Fst<Arc> &fst,
                                      float delta = kShortestDelta) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  std::vector<Weight> distance;
  if (Weight::Properties() & kRightSemiring) {
    ShortestDistance(fst, &distance, false, delta);
    if (distance.size() == 1 && !distance[0].Member()) {
      return Arc::Weight::NoWeight();
    }
    Adder<Weight> adder;  // maintains cumulative sum accurately
    for (StateId state = 0; state < distance.size(); ++state) {
      adder.Add(Times(distance[state], fst.Final(state)));
    }
    return adder.Sum();
  } else {
    ShortestDistance(fst, &distance, true, delta);
    const auto state = fst.Start();
    if (distance.size() == 1 && !distance[0].Member()) {
      return Arc::Weight::NoWeight();
    }
    return state != kNoStateId && state < distance.size() ? distance[state]
                                                          : Weight::Zero();
  }
}

}  // namespace fst

#endif  // FST_SHORTEST_DISTANCE_H_
