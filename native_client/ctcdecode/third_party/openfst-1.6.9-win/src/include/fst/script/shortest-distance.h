// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_SHORTEST_DISTANCE_H_
#define FST_SCRIPT_SHORTEST_DISTANCE_H_

#include <tuple>
#include <vector>

#include <fst/queue.h>
#include <fst/shortest-distance.h>
#include <fst/script/fst-class.h>
#include <fst/script/prune.h>
#include <fst/script/script-impl.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

enum ArcFilterType {
  ANY_ARC_FILTER,
  EPSILON_ARC_FILTER,
  INPUT_EPSILON_ARC_FILTER,
  OUTPUT_EPSILON_ARC_FILTER
};

struct ShortestDistanceOptions {
  const QueueType queue_type;
  const ArcFilterType arc_filter_type;
  const int64_t source;
  const float delta;

  ShortestDistanceOptions(QueueType queue_type, ArcFilterType arc_filter_type,
                          int64_t source, float delta)
      : queue_type(queue_type),
        arc_filter_type(arc_filter_type),
        source(source),
        delta(delta) {}
};

namespace internal {

// Code to implement switching on queue and arc filter types.

template <class Arc, class Queue, class ArcFilter>
struct QueueConstructor {
  using Weight = typename Arc::Weight;

  static Queue *Construct(const Fst<Arc> &, const std::vector<Weight> *) {
    return new Queue();
  }
};

// Specializations to support queues with different constructors.

template <class Arc, class ArcFilter>
struct QueueConstructor<Arc, AutoQueue<typename Arc::StateId>, ArcFilter> {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  //  template<class Arc, class ArcFilter>
  static AutoQueue<StateId> *Construct(const Fst<Arc> &fst,
                                       const std::vector<Weight> *distance) {
    return new AutoQueue<StateId>(fst, distance, ArcFilter());
  }
};

template <class Arc, class ArcFilter>
struct QueueConstructor<
    Arc, NaturalShortestFirstQueue<typename Arc::StateId, typename Arc::Weight>,
    ArcFilter> {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  static NaturalShortestFirstQueue<StateId, Weight> *Construct(
      const Fst<Arc> &, const std::vector<Weight> *distance) {
    return new NaturalShortestFirstQueue<StateId, Weight>(*distance);
  }
};

template <class Arc, class ArcFilter>
struct QueueConstructor<Arc, TopOrderQueue<typename Arc::StateId>, ArcFilter> {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  static TopOrderQueue<StateId> *Construct(const Fst<Arc> &fst,
                                           const std::vector<Weight> *) {
    return new TopOrderQueue<StateId>(fst, ArcFilter());
  }
};

template <class Arc, class Queue, class ArcFilter>
void ShortestDistance(const Fst<Arc> &fst,
                      std::vector<typename Arc::Weight> *distance,
                      const ShortestDistanceOptions &opts) {
  std::unique_ptr<Queue> queue(
      QueueConstructor<Arc, Queue, ArcFilter>::Construct(fst, distance));
  const fst::ShortestDistanceOptions<Arc, Queue, ArcFilter> sopts(
      queue.get(), ArcFilter(), opts.source, opts.delta);
  ShortestDistance(fst, distance, sopts);
}

template <class Arc, class Queue>
void ShortestDistance(const Fst<Arc> &fst,
                      std::vector<typename Arc::Weight> *distance,
                      const ShortestDistanceOptions &opts) {
  switch (opts.arc_filter_type) {
    case ANY_ARC_FILTER: {
      ShortestDistance<Arc, Queue, AnyArcFilter<Arc>>(fst, distance, opts);
      return;
    }
    case EPSILON_ARC_FILTER: {
      ShortestDistance<Arc, Queue, EpsilonArcFilter<Arc>>(fst, distance, opts);
      return;
    }
    case INPUT_EPSILON_ARC_FILTER: {
      ShortestDistance<Arc, Queue, InputEpsilonArcFilter<Arc>>(fst, distance,
                                                               opts);
      return;
    }
    case OUTPUT_EPSILON_ARC_FILTER: {
      ShortestDistance<Arc, Queue, OutputEpsilonArcFilter<Arc>>(fst, distance,
                                                                opts);
      return;
    }
    default: {
      FSTERROR() << "ShortestDistance: Unknown arc filter type: "
                 << opts.arc_filter_type;
      distance->clear();
      distance->resize(1, Arc::Weight::NoWeight());
      return;
    }
  }
}

}  // namespace internal

using ShortestDistanceArgs1 =
    std::tuple<const FstClass &, std::vector<WeightClass> *,
               const ShortestDistanceOptions &>;

template <class Arc>
void ShortestDistance(ShortestDistanceArgs1 *args) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  const auto &opts = std::get<2>(*args);
  std::vector<Weight> typed_distance;
  switch (opts.queue_type) {
    case AUTO_QUEUE: {
      internal::ShortestDistance<Arc, AutoQueue<StateId>>(fst, &typed_distance,
                                                          opts);
      break;
    }
    case FIFO_QUEUE: {
      internal::ShortestDistance<Arc, FifoQueue<StateId>>(fst, &typed_distance,
                                                          opts);
      break;
    }
    case LIFO_QUEUE: {
      internal::ShortestDistance<Arc, LifoQueue<StateId>>(fst, &typed_distance,
                                                          opts);
      break;
    }
    case SHORTEST_FIRST_QUEUE: {
      internal::ShortestDistance<Arc,
                                 NaturalShortestFirstQueue<StateId, Weight>>(
          fst, &typed_distance, opts);
      break;
    }
    case STATE_ORDER_QUEUE: {
      internal::ShortestDistance<Arc, StateOrderQueue<StateId>>(
          fst, &typed_distance, opts);
      break;
    }
    case TOP_ORDER_QUEUE: {
      internal::ShortestDistance<Arc, TopOrderQueue<StateId>>(
          fst, &typed_distance, opts);
      break;
    }
    default: {
      FSTERROR() << "ShortestDistance: Unknown queue type: " << opts.queue_type;
      typed_distance.clear();
      typed_distance.resize(1, Arc::Weight::NoWeight());
      break;
    }
  }
  internal::CopyWeights(typed_distance, std::get<1>(*args));
}

using ShortestDistanceArgs2 =
    std::tuple<const FstClass &, std::vector<WeightClass> *, bool, double>;

template <class Arc>
void ShortestDistance(ShortestDistanceArgs2 *args) {
  using Weight = typename Arc::Weight;
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  std::vector<Weight> typed_distance;
  ShortestDistance(fst, &typed_distance, std::get<2>(*args),
                   std::get<3>(*args));
  internal::CopyWeights(typed_distance, std::get<1>(*args));
}

void ShortestDistance(const FstClass &fst, std::vector<WeightClass> *distance,
                      const ShortestDistanceOptions &opts);

void ShortestDistance(const FstClass &ifst, std::vector<WeightClass> *distance,
                      bool reverse = false,
                      double delta = fst::kShortestDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_SHORTEST_DISTANCE_H_
