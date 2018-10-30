// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>
#include <fst/fst.h>
#include <fst/weight.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_string(weight, "", "Weight threshold");
DEFINE_int64(nstate, fst::kNoStateId, "State number threshold");
DEFINE_int64(subsequential_label, 0,
             "Input label of arc corresponding to residual final output when"
             " producing a subsequential transducer");
DEFINE_string(det_type, "functional",
              "Type of determinization: \"functional\", "
              "\"nonfunctional\", \"disambiguate\"");
DEFINE_bool(increment_subsequential_label, false,
            "Increment subsequential_label to obtain distinct labels for "
            " subsequential arcs at a given state");

int fstdeterminize_main(int argc, char **argv);

int main(int argc, char **argv) { return fstdeterminize_main(argc, argv); }
