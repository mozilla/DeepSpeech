// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>
#include <fst/weight.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_string(map_type, "identity",
              "Map operation, one of: \"arc_sum\", \"arc_unique\", "
              "\"float_power\" (--power)\", \"identity\", \"input_epsilon\", "
              "\"invert\", \"output_epsilon\", \"plus (--weight)\", "
              "\"quantize (--delta)\", \"rmweight\", \"superfinal\", "
              "\"power (--power)\", \"times (--weight)\", \"to_log\", "
              "\"to_log64\", \"to_std\"");
DEFINE_double(power, 1.0, "Power parameter");
DEFINE_string(weight, "", "Weight parameter");

int fstmap_main(int argc, char **argv);

int main(int argc, char **argv) { return fstmap_main(argc, argv); }
