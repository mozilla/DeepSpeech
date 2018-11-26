// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_string(call_arc_labeling, "input",
              "Which labels to make non-epsilon on the call arc. "
              "One of: \"input\" (default), \"output\", \"both\", \"neither\"");
DEFINE_string(return_arc_labeling, "neither",
              "Which labels to make non-epsilon on the return arc. "
              "One of: \"input\", \"output\", \"both\", \"neither\" (default)");
DEFINE_int64(return_label, 0, "Label to put on return arc");
DEFINE_bool(epsilon_on_replace, false, "Call/return arcs are epsilon arcs?");

int fstreplace_main(int argc, char **argv);

int main(int argc, char **argv) { return fstreplace_main(argc, argv); }
