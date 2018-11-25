// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>
#include <fst/fst.h>
#include <fst/shortest-distance.h>
#include <fst/weight.h>

DEFINE_double(delta, fst::kShortestDelta, "Comparison/quantization delta");
DEFINE_int32(nshortest, 1, "Return N-shortest paths");
DEFINE_int64(nstate, fst::kNoStateId, "State number threshold");
DEFINE_string(queue_type, "auto",
              "Queue type: one of \"auto\", "
              "\"fifo\", \"lifo\", \"shortest\', \"state\", \"top\"");
DEFINE_bool(unique, false, "Return unique strings");
DEFINE_string(weight, "", "Weight threshold");

int fstshortestpath_main(int argc, char **argv);

int main(int argc, char **argv) { return fstshortestpath_main(argc, argv); }
