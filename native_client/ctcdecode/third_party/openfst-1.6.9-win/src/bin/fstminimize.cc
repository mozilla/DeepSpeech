// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>
#include <fst/shortest-distance.h>
#include <fst/weight.h>

DEFINE_double(delta, fst::kShortestDelta, "Comparison/quantization delta");
DEFINE_bool(allow_nondet, false, "Minimize non-deterministic FSTs");

int fstminimize_main(int argc, char **argv);

int main(int argc, char **argv) { return fstminimize_main(argc, argv); }
