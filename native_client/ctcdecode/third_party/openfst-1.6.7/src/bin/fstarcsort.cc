// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/compat.h>
#include <fst/flags.h>

DEFINE_string(sort_type, "ilabel",
              "Comparison method, one of: \"ilabel\", \"olabel\"");

int fstarcsort_main(int argc, char **argv);

int main(int argc, char **argv) { return fstarcsort_main(argc, argv); }
