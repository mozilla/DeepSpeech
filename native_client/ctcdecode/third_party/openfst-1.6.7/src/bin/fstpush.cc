// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>
#include <fst/weight.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_bool(push_weights, false, "Push weights");
DEFINE_bool(push_labels, false, "Push output labels");
DEFINE_bool(remove_total_weight, false,
            "Remove total weight when pushing weights");
DEFINE_bool(remove_common_affix, false,
            "Remove common prefix/suffix when pushing labels");
DEFINE_bool(to_final, false, "Push/reweight to final (vs. to initial) states");

int fstpush_main(int argc, char **argv);

int main(int argc, char **argv) { return fstpush_main(argc, argv); }
