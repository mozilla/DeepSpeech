// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(to_final, false, "Push/reweight to final (vs. to initial) states");

int fstreweight_main(int argc, char **argv);

int main(int argc, char **argv) { return fstreweight_main(argc, argv); }
