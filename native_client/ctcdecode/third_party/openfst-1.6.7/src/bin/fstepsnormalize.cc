// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(eps_norm_output, false, "Normalize output epsilons");

int fstepsnormalize_main(int argc, char **argv);

int main(int argc, char **argv) { return fstepsnormalize_main(argc, argv); }
