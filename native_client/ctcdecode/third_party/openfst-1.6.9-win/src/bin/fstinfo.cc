// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_string(arc_filter, "any",
              "Arc filter: one of:"
              " \"any\", \"epsilon\", \"iepsilon\", \"oepsilon\"; "
              "this only affects the counts of (co)accessible states, "
              "connected states, and (strongly) connected components");
DEFINE_string(info_type, "auto",
              "Info format: one of: \"auto\", \"long\", \"short\"");
DEFINE_bool(pipe, false, "Send info to stderr, input to stdout");
DEFINE_bool(test_properties, true,
            "Compute property values (if unknown to FST)");
DEFINE_bool(fst_verify, true, "Verify FST sanity");

int fstinfo_main(int argc, char **argv);

int main(int argc, char **argv) { return fstinfo_main(argc, argv); }
