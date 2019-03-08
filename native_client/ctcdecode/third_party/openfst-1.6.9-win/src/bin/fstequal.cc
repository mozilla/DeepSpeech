// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>
#include <fst/weight.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");

int fstequal_main(int argc, char **argv);

int main(int argc, char **argv) { return fstequal_main(argc, argv); }
