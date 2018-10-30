// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/shortest-distance.h>

namespace fst {
namespace script {

void ShortestDistance(const FstClass &fst, std::vector<WeightClass> *distance,
                      const ShortestDistanceOptions &opts) {
  ShortestDistanceArgs1 args(fst, distance, opts);
  Apply<Operation<ShortestDistanceArgs1>>("ShortestDistance", fst.ArcType(),
                                          &args);
}

void ShortestDistance(const FstClass &ifst, std::vector<WeightClass> *distance,
                      bool reverse, double delta) {
  ShortestDistanceArgs2 args(ifst, distance, reverse, delta);
  Apply<Operation<ShortestDistanceArgs2>>("ShortestDistance", ifst.ArcType(),
                                          &args);
}

REGISTER_FST_OPERATION(ShortestDistance, StdArc, ShortestDistanceArgs1);
REGISTER_FST_OPERATION(ShortestDistance, LogArc, ShortestDistanceArgs1);
REGISTER_FST_OPERATION(ShortestDistance, Log64Arc, ShortestDistanceArgs1);

REGISTER_FST_OPERATION(ShortestDistance, StdArc, ShortestDistanceArgs2);
REGISTER_FST_OPERATION(ShortestDistance, LogArc, ShortestDistanceArgs2);
REGISTER_FST_OPERATION(ShortestDistance, Log64Arc, ShortestDistanceArgs2);

}  // namespace script
}  // namespace fst
