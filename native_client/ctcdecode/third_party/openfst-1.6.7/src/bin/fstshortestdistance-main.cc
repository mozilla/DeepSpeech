// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Find shortest distances in an FST.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/shortest-distance.h>
#include <fst/script/text-io.h>

DECLARE_bool(reverse);
DECLARE_double(delta);
DECLARE_int64(nstate);
DECLARE_string(queue_type);

int fstshortestdistance_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::WeightClass;
  using fst::QueueType;
  using fst::AUTO_QUEUE;

  string usage = "Finds shortest distance(s) in an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [distance.txt]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  string in_name = (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  std::vector<WeightClass> distance;

  QueueType queue_type;
  if (!s::GetQueueType(FLAGS_queue_type, &queue_type)) {
    LOG(ERROR) << argv[0]
               << ": Unknown or unsupported queue type: " << FLAGS_queue_type;
    return 1;
  }

  if (FLAGS_reverse && queue_type != AUTO_QUEUE) {
    LOG(ERROR) << argv[0] << ": Can't use non-default queue with reverse";
    return 1;
  }

  if (FLAGS_reverse) {
    s::ShortestDistance(*ifst, &distance, FLAGS_reverse, FLAGS_delta);
  } else {
    const s::ShortestDistanceOptions opts(queue_type, s::ANY_ARC_FILTER,
                                          FLAGS_nstate, FLAGS_delta);
    s::ShortestDistance(*ifst, &distance, opts);
  }

  return !s::WritePotentials(out_name, distance);
}
