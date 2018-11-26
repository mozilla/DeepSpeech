// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes for various FST state queues with a unified interface.

#ifndef FST_QUEUE_H_
#define FST_QUEUE_H_

#include <deque>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/arcfilter.h>
#include <fst/connect.h>
#include <fst/heap.h>
#include <fst/topsort.h>


namespace fst {

// The Queue interface is:
//
// template <class S>
// class Queue {
//  public:
//   using StateId = S;
//
//   // Constructor: may need args (e.g., FST, comparator) for some queues.
//   Queue(...) override;
//
//   // Returns the head of the queue.
//   StateId Head() const override;
//
//   // Inserts a state.
//   void Enqueue(StateId s) override;
//
//   // Removes the head of the queue.
//   void Dequeue() override;
//
//   // Updates ordering of state s when weight changes, if necessary.
//   void Update(StateId s) override;
//
//   // Is the queue empty?
//   bool Empty() const override;
//
//   // Removes all states from the queue.
//   void Clear() override;
// };

// State queue types.
enum QueueType {
  TRIVIAL_QUEUE = 0,         // Single state queue.
  FIFO_QUEUE = 1,            // First-in, first-out queue.
  LIFO_QUEUE = 2,            // Last-in, first-out queue.
  SHORTEST_FIRST_QUEUE = 3,  // Shortest-first queue.
  TOP_ORDER_QUEUE = 4,       // Topologically-ordered queue.
  STATE_ORDER_QUEUE = 5,     // State ID-ordered queue.
  SCC_QUEUE = 6,             // Component graph top-ordered meta-queue.
  AUTO_QUEUE = 7,            // Auto-selected queue.
  OTHER_QUEUE = 8
};

// QueueBase, templated on the StateId, is a virtual base class shared by all
// queues considered by AutoQueue.
template <class S>
class QueueBase {
 public:
  using StateId = S;

  virtual ~QueueBase() {}

  // Concrete implementation.

  explicit QueueBase(QueueType type) : queue_type_(type), error_(false) {}

  void SetError(bool error) { error_ = error; }

  bool Error() const { return error_; }

  QueueType Type() const { return queue_type_; }

  // Virtual interface.

  virtual StateId Head() const = 0;
  virtual void Enqueue(StateId) = 0;
  virtual void Dequeue() = 0;
  virtual void Update(StateId) = 0;
  virtual bool Empty() const = 0;
  virtual void Clear() = 0;

 private:
  QueueType queue_type_;
  bool error_;
};

// Trivial queue discipline; one may enqueue at most one state at a time. It
// can be used for strongly connected components with only one state and no
// self-loops.
template <class S>
class TrivialQueue : public QueueBase<S> {
 public:
  using StateId = S;

  TrivialQueue() : QueueBase<StateId>(TRIVIAL_QUEUE), front_(kNoStateId) {}

  virtual ~TrivialQueue() = default;

  StateId Head() const final { return front_; }

  void Enqueue(StateId s) final { front_ = s; }

  void Dequeue() final { front_ = kNoStateId; }

  void Update(StateId) final {}

  bool Empty() const final { return front_ == kNoStateId; }

  void Clear() final { front_ = kNoStateId; }

 private:
  StateId front_;
};

// First-in, first-out queue discipline.
//
// This is not a final class.
template <class S>
class FifoQueue : public QueueBase<S> {
 public:
  using StateId = S;

  FifoQueue() : QueueBase<StateId>(FIFO_QUEUE) {}

  virtual ~FifoQueue() = default;

  StateId Head() const override { return queue_.back(); }

  void Enqueue(StateId s) override { queue_.push_front(s); }

  void Dequeue() override { queue_.pop_back(); }

  void Update(StateId) override {}

  bool Empty() const override { return queue_.empty(); }

  void Clear() override { queue_.clear(); }

 private:
  std::deque<StateId> queue_;
};

// Last-in, first-out queue discipline.
template <class S>
class LifoQueue : public QueueBase<S> {
 public:
  using StateId = S;

  LifoQueue() : QueueBase<StateId>(LIFO_QUEUE) {}

  virtual ~LifoQueue() = default;

  StateId Head() const final { return queue_.front(); }

  void Enqueue(StateId s) final { queue_.push_front(s); }

  void Dequeue() final { queue_.pop_front(); }

  void Update(StateId) final {}

  bool Empty() const final { return queue_.empty(); }

  void Clear() final { queue_.clear(); }

 private:
  std::deque<StateId> queue_;
};

// Shortest-first queue discipline, templated on the StateId and as well as a
// comparison functor used to compare two StateIds. If a (single) state's order
// changes, it can be reordered in the queue with a call to Update(). If update
// is false, call to Update() does not reorder the queue.
//
// This is not a final class.
template <typename S, typename Compare, bool update = true>
class ShortestFirstQueue : public QueueBase<S> {
 public:
  using StateId = S;

  explicit ShortestFirstQueue(Compare comp)
      : QueueBase<StateId>(SHORTEST_FIRST_QUEUE), heap_(comp) {}

  virtual ~ShortestFirstQueue() = default;

  StateId Head() const override { return heap_.Top(); }

  void Enqueue(StateId s) override {
    if (update) {
      for (StateId i = key_.size(); i <= s; ++i) key_.push_back(kNoStateId);
      key_[s] = heap_.Insert(s);
    } else {
      heap_.Insert(s);
    }
  }

  void Dequeue() override {
    if (update) {
      key_[heap_.Pop()] = kNoStateId;
    } else {
      heap_.Pop();
    }
  }

  void Update(StateId s) override {
    if (!update) return;
    if (s >= key_.size() || key_[s] == kNoStateId) {
      Enqueue(s);
    } else {
      heap_.Update(key_[s], s);
    }
  }

  bool Empty() const override { return heap_.Empty(); }

  void Clear() override {
    heap_.Clear();
    if (update) key_.clear();
  }

  const Compare &GetCompare() const { return heap_.GetCompare(); }

 private:
  Heap<StateId, Compare> heap_;
  std::vector<ssize_t> key_;
};

namespace internal {

// Given a vector that maps from states to weights, and a comparison functor
// for weights, this class defines a comparison function object between states.
template <typename StateId, typename Less>
class StateWeightCompare {
 public:
  using Weight = typename Less::Weight;

  StateWeightCompare(const std::vector<Weight> &weights, const Less &less)
      : weights_(weights), less_(less) {}

  bool operator()(const StateId s1, const StateId s2) const {
    return less_(weights_[s1], weights_[s2]);
  }

 private:
  // Borrowed references.
  const std::vector<Weight> &weights_;
  const Less &less_;
};

}  // namespace internal

// Shortest-first queue discipline, templated on the StateId and Weight, is
// specialized to use the weight's natural order for the comparison function.
template <typename S, typename Weight>
class NaturalShortestFirstQueue final
    : public ShortestFirstQueue<
          S, internal::StateWeightCompare<S, NaturalLess<Weight>>> {
 public:
  using StateId = S;
  using Compare = internal::StateWeightCompare<StateId, NaturalLess<Weight>>;

  explicit NaturalShortestFirstQueue(const std::vector<Weight> &distance)
      : ShortestFirstQueue<StateId, Compare>(Compare(distance, less_)) {}

  virtual ~NaturalShortestFirstQueue() = default;

 private:
  // This is non-static because the constructor for non-idempotent weights will
  // result in a an error.
  const NaturalLess<Weight> less_{};
};

// Topological-order queue discipline, templated on the StateId. States are
// ordered in the queue topologically. The FST must be acyclic.
template <class S>
class TopOrderQueue : public QueueBase<S> {
 public:
  using StateId = S;

  // This constructor computes the topological order. It accepts an arc filter
  // to limit the transitions considered in that computation (e.g., only the
  // epsilon graph).
  template <class Arc, class ArcFilter>
  TopOrderQueue(const Fst<Arc> &fst, ArcFilter filter)
      : QueueBase<StateId>(TOP_ORDER_QUEUE),
        front_(0),
        back_(kNoStateId),
        order_(0),
        state_(0) {
    bool acyclic;
    TopOrderVisitor<Arc> top_order_visitor(&order_, &acyclic);
    DfsVisit(fst, &top_order_visitor, filter);
    if (!acyclic) {
      FSTERROR() << "TopOrderQueue: FST is not acyclic";
      QueueBase<S>::SetError(true);
    }
    state_.resize(order_.size(), kNoStateId);
  }

  // This constructor is passed the pre-computed topological order.
  explicit TopOrderQueue(const std::vector<StateId> &order)
      : QueueBase<StateId>(TOP_ORDER_QUEUE),
        front_(0),
        back_(kNoStateId),
        order_(order),
        state_(order.size(), kNoStateId) {}

  virtual ~TopOrderQueue() = default;

  StateId Head() const final { return state_[front_]; }

  void Enqueue(StateId s) final {
    if (front_ > back_) {
      front_ = back_ = order_[s];
    } else if (order_[s] > back_) {
      back_ = order_[s];
    } else if (order_[s] < front_) {
      front_ = order_[s];
    }
    state_[order_[s]] = s;
  }

  void Dequeue() final {
    state_[front_] = kNoStateId;
    while ((front_ <= back_) && (state_[front_] == kNoStateId)) ++front_;
  }

  void Update(StateId) final {}

  bool Empty() const final { return front_ > back_; }

  void Clear() final {
    for (StateId s = front_; s <= back_; ++s) state_[s] = kNoStateId;
    back_ = kNoStateId;
    front_ = 0;
  }

 private:
  StateId front_;
  StateId back_;
  std::vector<StateId> order_;
  std::vector<StateId> state_;
};

// State order queue discipline, templated on the StateId. States are ordered in
// the queue by state ID.
template <class S>
class StateOrderQueue : public QueueBase<S> {
 public:
  using StateId = S;

  StateOrderQueue()
      : QueueBase<StateId>(STATE_ORDER_QUEUE), front_(0), back_(kNoStateId) {}

  virtual ~StateOrderQueue() = default;

  StateId Head() const final { return front_; }

  void Enqueue(StateId s) final {
    if (front_ > back_) {
      front_ = back_ = s;
    } else if (s > back_) {
      back_ = s;
    } else if (s < front_) {
      front_ = s;
    }
    while (enqueued_.size() <= s) enqueued_.push_back(false);
    enqueued_[s] = true;
  }

  void Dequeue() final {
    enqueued_[front_] = false;
    while ((front_ <= back_) && (enqueued_[front_] == false)) ++front_;
  }

  void Update(StateId) final {}

  bool Empty() const final { return front_ > back_; }

  void Clear() final {
    for (StateId i = front_; i <= back_; ++i) enqueued_[i] = false;
    front_ = 0;
    back_ = kNoStateId;
  }

 private:
  StateId front_;
  StateId back_;
  std::vector<bool> enqueued_;
};

// SCC topological-order meta-queue discipline, templated on the StateId and a
// queue used inside each SCC. It visits the SCCs of an FST in topological
// order. Its constructor is passed the queues to to use within an SCC.
template <class S, class Queue>
class SccQueue : public QueueBase<S> {
 public:
  using StateId = S;

  // Constructor takes a vector specifying the SCC number per state and a
  // vector giving the queue to use per SCC number.
  SccQueue(const std::vector<StateId> &scc,
           std::vector<std::unique_ptr<Queue>> *queue)
      : QueueBase<StateId>(SCC_QUEUE),
        queue_(queue),
        scc_(scc),
        front_(0),
        back_(kNoStateId) {}

  virtual ~SccQueue() = default;

  StateId Head() const final {
    while ((front_ <= back_) &&
           (((*queue_)[front_] && (*queue_)[front_]->Empty()) ||
            (((*queue_)[front_] == nullptr) &&
             ((front_ >= trivial_queue_.size()) ||
              (trivial_queue_[front_] == kNoStateId))))) {
      ++front_;
    }
    if ((*queue_)[front_]) {
      return (*queue_)[front_]->Head();
    } else {
      return trivial_queue_[front_];
    }
  }

  void Enqueue(StateId s) final {
    if (front_ > back_) {
      front_ = back_ = scc_[s];
    } else if (scc_[s] > back_) {
      back_ = scc_[s];
    } else if (scc_[s] < front_) {
      front_ = scc_[s];
    }
    if ((*queue_)[scc_[s]]) {
      (*queue_)[scc_[s]]->Enqueue(s);
    } else {
      while (trivial_queue_.size() <= scc_[s]) {
        trivial_queue_.push_back(kNoStateId);
      }
      trivial_queue_[scc_[s]] = s;
    }
  }

  void Dequeue() final {
    if ((*queue_)[front_]) {
      (*queue_)[front_]->Dequeue();
    } else if (front_ < trivial_queue_.size()) {
      trivial_queue_[front_] = kNoStateId;
    }
  }

  void Update(StateId s) final {
    if ((*queue_)[scc_[s]]) (*queue_)[scc_[s]]->Update(s);
  }

  bool Empty() const final {
    // Queues SCC number back_ is not empty unless back_ == front_.
    if (front_ < back_) {
      return false;
    } else if (front_ > back_) {
      return true;
    } else if ((*queue_)[front_]) {
      return (*queue_)[front_]->Empty();
    } else {
      return (front_ >= trivial_queue_.size()) ||
             (trivial_queue_[front_] == kNoStateId);
    }
  }

  void Clear() final {
    for (StateId i = front_; i <= back_; ++i) {
      if ((*queue_)[i]) {
        (*queue_)[i]->Clear();
      } else if (i < trivial_queue_.size()) {
        trivial_queue_[i] = kNoStateId;
      }
    }
    front_ = 0;
    back_ = kNoStateId;
  }

 private:
  std::vector<std::unique_ptr<Queue>> *queue_;
  const std::vector<StateId> &scc_;
  mutable StateId front_;
  StateId back_;
  std::vector<StateId> trivial_queue_;
};

// Automatic queue discipline. It selects a queue discipline for a given FST
// based on its properties.
template <class S>
class AutoQueue : public QueueBase<S> {
 public:
  using StateId = S;

  // This constructor takes a state distance vector that, if non-null and if
  // the Weight type has the path property, will entertain the shortest-first
  // queue using the natural order w.r.t to the distance.
  template <class Arc, class ArcFilter>
  AutoQueue(const Fst<Arc> &fst,
            const std::vector<typename Arc::Weight> *distance, ArcFilter filter)
      : QueueBase<StateId>(AUTO_QUEUE) {
    using Weight = typename Arc::Weight;
    using Less = NaturalLess<Weight>;
    using Compare = internal::StateWeightCompare<StateId, Less>;
    // First checks if the FST is known to have these properties.
    const auto props =
        fst.Properties(kAcyclic | kCyclic | kTopSorted | kUnweighted, false);
    if ((props & kTopSorted) || fst.Start() == kNoStateId) {
      queue_.reset(new StateOrderQueue<StateId>());
      VLOG(2) << "AutoQueue: using state-order discipline";
    } else if (props & kAcyclic) {
      queue_.reset(new TopOrderQueue<StateId>(fst, filter));
      VLOG(2) << "AutoQueue: using top-order discipline";
    } else if ((props & kUnweighted) && (Weight::Properties() & kIdempotent)) {
      queue_.reset(new LifoQueue<StateId>());
      VLOG(2) << "AutoQueue: using LIFO discipline";
    } else {
      uint64 properties;
      // Decomposes into strongly-connected components.
      SccVisitor<Arc> scc_visitor(&scc_, nullptr, nullptr, &properties);
      DfsVisit(fst, &scc_visitor, filter);
      auto nscc = *std::max_element(scc_.begin(), scc_.end()) + 1;
      std::vector<QueueType> queue_types(nscc);
      std::unique_ptr<Less> less;
      std::unique_ptr<Compare> comp;
      if (distance && (Weight::Properties() & kPath) == kPath) {
        less.reset(new Less);
        comp.reset(new Compare(*distance, *less));
      }
      // Finds the queue type to use per SCC.
      bool unweighted;
      bool all_trivial;
      SccQueueType(fst, scc_, &queue_types, filter, less.get(), &all_trivial,
                   &unweighted);
      // If unweighted and semiring is idempotent, uses LIFO queue.
      if (unweighted) {
        queue_.reset(new LifoQueue<StateId>());
        VLOG(2) << "AutoQueue: using LIFO discipline";
        return;
      }
      // If all the SCC are trivial, the FST is acyclic and the scc number gives
      // the topological order.
      if (all_trivial) {
        queue_.reset(new TopOrderQueue<StateId>(scc_));
        VLOG(2) << "AutoQueue: using top-order discipline";
        return;
      }
      VLOG(2) << "AutoQueue: using SCC meta-discipline";
      queues_.resize(nscc);
      for (StateId i = 0; i < nscc; ++i) {
        switch (queue_types[i]) {
          case TRIVIAL_QUEUE:
            queues_[i].reset();
            VLOG(3) << "AutoQueue: SCC #" << i << ": using trivial discipline";
            break;
          case SHORTEST_FIRST_QUEUE:
            queues_[i].reset(
                new ShortestFirstQueue<StateId, Compare, false>(*comp));
            VLOG(3) << "AutoQueue: SCC #" << i
                    << ": using shortest-first discipline";
            break;
          case LIFO_QUEUE:
            queues_[i].reset(new LifoQueue<StateId>());
            VLOG(3) << "AutoQueue: SCC #" << i << ": using LIFO discipline";
            break;
          case FIFO_QUEUE:
          default:
            queues_[i].reset(new FifoQueue<StateId>());
            VLOG(3) << "AutoQueue: SCC #" << i << ": using FIFO discipine";
            break;
        }
      }
      queue_.reset(new SccQueue<StateId, QueueBase<StateId>>(scc_, &queues_));
    }
  }

  virtual ~AutoQueue() = default;

  StateId Head() const final { return queue_->Head(); }

  void Enqueue(StateId s) final { queue_->Enqueue(s); }

  void Dequeue() final { queue_->Dequeue(); }

  void Update(StateId s) final { queue_->Update(s); }

  bool Empty() const final { return queue_->Empty(); }

  void Clear() final { queue_->Clear(); }

 private:
  template <class Arc, class ArcFilter, class Less>
  static void SccQueueType(const Fst<Arc> &fst, const std::vector<StateId> &scc,
                           std::vector<QueueType> *queue_types,
                           ArcFilter filter, Less *less, bool *all_trivial,
                           bool *unweighted);

  std::unique_ptr<QueueBase<StateId>> queue_;
  std::vector<std::unique_ptr<QueueBase<StateId>>> queues_;
  std::vector<StateId> scc_;
};

// Examines the states in an FST's strongly connected components and determines
// which type of queue to use per SCC. Stores result as a vector of QueueTypes
// which is assumed to have length equal to the number of SCCs. An arc filter
// is used to limit the transitions considered (e.g., only the epsilon graph).
// The argument all_trivial is set to true if every queue is the trivial queue.
// The argument unweighted is set to true if the semiring is idempotent and all
// the arc weights are equal to Zero() or One().
template <class StateId>
template <class Arc, class ArcFilter, class Less>
void AutoQueue<StateId>::SccQueueType(const Fst<Arc> &fst,
                                      const std::vector<StateId> &scc,
                                      std::vector<QueueType> *queue_type,
                                      ArcFilter filter, Less *less,
                                      bool *all_trivial, bool *unweighted) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  *all_trivial = true;
  *unweighted = true;
  for (StateId i = 0; i < queue_type->size(); ++i) {
    (*queue_type)[i] = TRIVIAL_QUEUE;
  }
  for (StateIterator<Fst<Arc>> sit(fst); !sit.Done(); sit.Next()) {
    const auto state = sit.Value();
    for (ArcIterator<Fst<Arc>> ait(fst, state); !ait.Done(); ait.Next()) {
      const auto &arc = ait.Value();
      if (!filter(arc)) continue;
      if (scc[state] == scc[arc.nextstate]) {
        auto &type = (*queue_type)[scc[state]];
        if (!less || ((*less)(arc.weight, Weight::One()))) {
          type = FIFO_QUEUE;
        } else if ((type == TRIVIAL_QUEUE) || (type == LIFO_QUEUE)) {
          if (!(Weight::Properties() & kIdempotent) ||
              (arc.weight != Weight::Zero() && arc.weight != Weight::One())) {
            type = SHORTEST_FIRST_QUEUE;
          } else {
            type = LIFO_QUEUE;
          }
        }
        if (type != TRIVIAL_QUEUE) *all_trivial = false;
      }
      if (!(Weight::Properties() & kIdempotent) ||
          (arc.weight != Weight::Zero() && arc.weight != Weight::One())) {
        *unweighted = false;
      }
    }
  }
}

// An A* estimate is a function object that maps from a state ID to a an
// estimate of the shortest distance to the final states.

// A trivial A* estimate, yielding a queue which behaves the same in Dijkstra's
// algorithm.
template <typename StateId, typename Weight>
struct TrivialAStarEstimate {
  const Weight &operator()(StateId) const { return Weight::One(); }
};

// A non-trivial A* estimate using a vector of the estimated future costs.
template <typename StateId, typename Weight>
class NaturalAStarEstimate {
 public:
  NaturalAStarEstimate(const std::vector<Weight> &beta) :
          beta_(beta) {}

  const Weight &operator()(StateId s) const { return beta_[s]; }

 private:
  const std::vector<Weight> &beta_;
};

// Given a vector that maps from states to weights representing the shortest
// distance from the initial state, a comparison function object between
// weights, and an estimate of the shortest distance to the final states, this
// class defines a comparison function object between states.
template <typename S, typename Less, typename Estimate>
class AStarWeightCompare {
 public:
  using StateId = S;
  using Weight = typename Less::Weight;

  AStarWeightCompare(const std::vector<Weight> &weights, const Less &less,
                     const Estimate &estimate)
      : weights_(weights), less_(less), estimate_(estimate) {}

  bool operator()(StateId s1, StateId s2) const {
    const auto w1 = Times(weights_[s1], estimate_(s1));
    const auto w2 = Times(weights_[s2], estimate_(s2));
    return less_(w1, w2);
  }

  const Estimate &GetEstimate() const { return estimate_; }

 private:
  const std::vector<Weight> &weights_;
  const Less &less_;
  const Estimate &estimate_;
};

// A* queue discipline templated on StateId, Weight, and Estimate.
template <typename S, typename Weight, typename Estimate>
class NaturalAStarQueue : public ShortestFirstQueue<
          S, AStarWeightCompare<S, NaturalLess<Weight>, Estimate>> {
 public:
  using StateId = S;
  using Compare = AStarWeightCompare<StateId, NaturalLess<Weight>, Estimate>;

  NaturalAStarQueue(const std::vector<Weight> &distance,
                    const Estimate &estimate)
      : ShortestFirstQueue<StateId, Compare>(
            Compare(distance, less_, estimate)) {}

  ~NaturalAStarQueue() = default;

 private:
  // This is non-static because the constructor for non-idempotent weights will
  // result in a an error.
  const NaturalLess<Weight> less_{};
};

// A state equivalence class is a function object that maps from a state ID to
// an equivalence class (state) ID. The trivial equivalence class maps a state
// ID to itself.
template <typename StateId>
struct TrivialStateEquivClass {
  StateId operator()(StateId s) const { return s; }
};

// Distance-based pruning queue discipline: Enqueues a state only when its
// shortest distance (so far), as specified by distance, is less than (as
// specified by comp) the shortest distance Times() the threshold to any state
// in the same equivalence class, as specified by the functor class_func. The
// underlying queue discipline is specified by queue. The ownership of queue is
// given to this class.
//
// This is not a final class.
template <typename Queue, typename Less, typename ClassFnc>
class PruneQueue : public QueueBase<typename Queue::StateId> {
 public:
  using StateId = typename Queue::StateId;
  using Weight = typename Less::Weight;

  PruneQueue(const std::vector<Weight> &distance, Queue *queue,
             const Less &less, const ClassFnc &class_fnc, Weight threshold)
      : QueueBase<StateId>(OTHER_QUEUE),
        distance_(distance),
        queue_(queue),
        less_(less),
        class_fnc_(class_fnc),
        threshold_(std::move(threshold)) {}

  virtual ~PruneQueue() = default;

  StateId Head() const override { return queue_->Head(); }

  void Enqueue(StateId s) override {
    const auto c = class_fnc_(s);
    if (c >= class_distance_.size()) {
      class_distance_.resize(c + 1, Weight::Zero());
    }
    if (less_(distance_[s], class_distance_[c])) {
      class_distance_[c] = distance_[s];
    }
    // Enqueues only if below threshold limit.
    const auto limit = Times(class_distance_[c], threshold_);
    if (less_(distance_[s], limit)) queue_->Enqueue(s);
  }

  void Dequeue() override { queue_->Dequeue(); }

  void Update(StateId s) override {
    const auto c = class_fnc_(s);
    if (less_(distance_[s], class_distance_[c])) {
      class_distance_[c] = distance_[s];
    }
    queue_->Update(s);
  }

  bool Empty() const override { return queue_->Empty(); }

  void Clear() override { queue_->Clear(); }

 private:
  const std::vector<Weight> &distance_;  // Shortest distance to state.
  std::unique_ptr<Queue> queue_;
  const Less &less_;                    // Borrowed reference.
  const ClassFnc &class_fnc_;           // Equivalence class functor.
  Weight threshold_;                    // Pruning weight threshold.
  std::vector<Weight> class_distance_;  // Shortest distance to class.
};

// Pruning queue discipline (see above) using the weight's natural order for the
// comparison function. The ownership of the queue argument is given to this
// class.
template <typename Queue, typename Weight, typename ClassFnc>
class NaturalPruneQueue final
    : public PruneQueue<Queue, NaturalLess<Weight>, ClassFnc> {
 public:
  using StateId = typename Queue::StateId;

  NaturalPruneQueue(const std::vector<Weight> &distance, Queue *queue,
                    const ClassFnc &class_fnc, Weight threshold)
      : PruneQueue<Queue, NaturalLess<Weight>, ClassFnc>(
            distance, queue, NaturalLess<Weight>(), class_fnc, threshold) {}

  virtual ~NaturalPruneQueue() = default;
};

// Filter-based pruning queue discipline: enqueues a state only if allowed by
// the filter, specified by the state filter functor argument. The underlying
// queue discipline is specified by the queue argument. The ownership of the
// queue is given to this class.
template <typename Queue, typename Filter>
class FilterQueue : public QueueBase<typename Queue::StateId> {
 public:
  using StateId = typename Queue::StateId;

  FilterQueue(Queue *queue, const Filter &filter)
      : QueueBase<StateId>(OTHER_QUEUE), queue_(queue), filter_(filter) {}

  virtual ~FilterQueue() = default;

  StateId Head() const final { return queue_->Head(); }

  // Enqueues only if allowed by state filter.
  void Enqueue(StateId s) final {
    if (filter_(s)) queue_->Enqueue(s);
  }

  void Dequeue() final { queue_->Dequeue(); }

  void Update(StateId s) final {}

  bool Empty() const final { return queue_->Empty(); }

  void Clear() final { queue_->Clear(); }

 private:
  std::unique_ptr<Queue> queue_;
  const Filter &filter_;
};

}  // namespace fst

#endif  // FST_QUEUE_H_
