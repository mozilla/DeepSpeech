// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RMEPSILON_H_
#define FST_SCRIPT_RMEPSILON_H_

#include <utility>
#include <vector>

#include <fst/queue.h>
#include <fst/rmepsilon.h>
#include <fst/script/fst-class.h>
#include <fst/script/shortest-distance.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

struct RmEpsilonOptions : public ShortestDistanceOptions {
  const bool connect;
  const WeightClass &weight_threshold;
  const int64_t state_threshold;

  RmEpsilonOptions(QueueType queue_type, bool connect,
                   const WeightClass &weight_threshold,
                   int64_t state_threshold = kNoStateId, float delta = kDelta)
      : ShortestDistanceOptions(queue_type, EPSILON_ARC_FILTER, kNoStateId,
                                delta),
        connect(connect),
        weight_threshold(weight_threshold),
        state_threshold(state_threshold) {}
};

namespace internal {

// Code to implement switching on queue types.

template <class Arc, class Queue>
void RmEpsilon(MutableFst<Arc> *fst,
               std::vector<typename Arc::Weight> *distance,
               const RmEpsilonOptions &opts, Queue *queue) {
  using Weight = typename Arc::Weight;
  const fst::RmEpsilonOptions<Arc, Queue> ropts(
      queue, opts.delta, opts.connect,
      *opts.weight_threshold.GetWeight<Weight>(), opts.state_threshold);
  RmEpsilon(fst, distance, ropts);
}

template <class Arc>
void RmEpsilon(MutableFst<Arc> *fst, const RmEpsilonOptions &opts) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  std::vector<Weight> distance;
  switch (opts.queue_type) {
    case AUTO_QUEUE: {
      AutoQueue<StateId> queue(*fst, &distance, EpsilonArcFilter<Arc>());
      RmEpsilon(fst, &distance, opts, &queue);
      return;
    }
    case FIFO_QUEUE: {
      FifoQueue<StateId> queue;
      RmEpsilon(fst, &distance, opts, &queue);
      return;
    }
    case LIFO_QUEUE: {
      LifoQueue<StateId> queue;
      RmEpsilon(fst, &distance, opts, &queue);
      return;
    }
    case SHORTEST_FIRST_QUEUE: {
      NaturalShortestFirstQueue<StateId, Weight> queue(distance);
      RmEpsilon(fst, &distance, opts, &queue);
      return;
    }
    case STATE_ORDER_QUEUE: {
      StateOrderQueue<StateId> queue;
      RmEpsilon(fst, &distance, opts, &queue);
      return;
    }
    case TOP_ORDER_QUEUE: {
      TopOrderQueue<StateId> queue(*fst, EpsilonArcFilter<Arc>());
      internal::RmEpsilon(fst, &distance, opts, &queue);
      return;
    }
    default: {
      FSTERROR() << "RmEpsilon: Unknown queue type: " << opts.queue_type;
      fst->SetProperties(kError, kError);
      return;
    }
  }
}

}  // namespace internal

using RmEpsilonArgs = std::pair<MutableFstClass *, const RmEpsilonOptions &>;

template <class Arc>
void RmEpsilon(RmEpsilonArgs *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  const auto &opts = std::get<1>(*args);
  internal::RmEpsilon(fst, opts);
}

void RmEpsilon(MutableFstClass *fst, const RmEpsilonOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RMEPSILON_H_
