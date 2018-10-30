// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_SHORTEST_PATH_H_
#define FST_SCRIPT_SHORTEST_PATH_H_

#include <memory>
#include <vector>

#include <fst/shortest-path.h>
#include <fst/script/fst-class.h>
#include <fst/script/shortest-distance.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

// Slightly simplified interface: `has_distance` and `first_path` are disabled.

struct ShortestPathOptions : public ShortestDistanceOptions {
  const int32 nshortest;
  const bool unique;
  const WeightClass &weight_threshold;
  const int64 state_threshold;

  ShortestPathOptions(QueueType queue_type, int32 nshortest, bool unique,
                      float delta, const WeightClass &weight_threshold,
                      int64 state_threshold = kNoStateId)
      : ShortestDistanceOptions(queue_type, ANY_ARC_FILTER, kNoStateId, delta),
        nshortest(nshortest),
        unique(unique),
        weight_threshold(weight_threshold),
        state_threshold(state_threshold) {}
};

namespace internal {

// Code to implement switching on queue types.

template <class Arc, class Queue>
void ShortestPath(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                  std::vector<typename Arc::Weight> *distance,
                  const ShortestPathOptions &opts) {
  using ArcFilter = AnyArcFilter<Arc>;
  using Weight = typename Arc::Weight;
  const std::unique_ptr<Queue> queue(
      QueueConstructor<Arc, Queue, ArcFilter>::Construct(ifst, distance));
  const fst::ShortestPathOptions<Arc, Queue, ArcFilter> sopts(
      queue.get(), ArcFilter(), opts.nshortest, opts.unique,
      /* has_distance=*/false, opts.delta, /* first_path=*/false,
      *opts.weight_threshold.GetWeight<Weight>(), opts.state_threshold);
  ShortestPath(ifst, ofst, distance, sopts);
}

template <class Arc>
void ShortestPath(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                  const ShortestPathOptions &opts) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  std::vector<Weight> distance;
  switch (opts.queue_type) {
    case AUTO_QUEUE: {
      ShortestPath<Arc, AutoQueue<StateId>>(ifst, ofst, &distance, opts);
      return;
    }
    case FIFO_QUEUE: {
      ShortestPath<Arc, FifoQueue<StateId>>(ifst, ofst, &distance, opts);
      return;
    }
    case LIFO_QUEUE: {
      ShortestPath<Arc, LifoQueue<StateId>>(ifst, ofst, &distance, opts);
      return;
    }
    case SHORTEST_FIRST_QUEUE: {
      ShortestPath<Arc, NaturalShortestFirstQueue<StateId, Weight>>(ifst, ofst,
                                                                    &distance,
                                                                    opts);
      return;
    }
    case STATE_ORDER_QUEUE: {
      ShortestPath<Arc, StateOrderQueue<StateId>>(ifst, ofst, &distance, opts);
      return;
    }
    case TOP_ORDER_QUEUE: {
      ShortestPath<Arc, TopOrderQueue<StateId>>(ifst, ofst, &distance, opts);
      return;
    }
    default: {
      FSTERROR() << "ShortestPath: Unknown queue type: "
                 << opts.queue_type;
      ofst->SetProperties(kError, kError);
      return;
    }
  }
}

}  // namespace internal

using ShortestPathArgs = std::tuple<const FstClass &, MutableFstClass *,
                                    const ShortestPathOptions &>;

template <class Arc>
void ShortestPath(ShortestPathArgs *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  const ShortestPathOptions &opts = std::get<2>(*args);
  internal::ShortestPath(ifst, ofst, opts);
}

void ShortestPath(const FstClass &ifst, MutableFstClass *ofst,
                  const ShortestPathOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_SHORTEST_PATH_H_
