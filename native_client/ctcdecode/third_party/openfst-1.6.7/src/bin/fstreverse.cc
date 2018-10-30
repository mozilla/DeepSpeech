// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(require_superinitial, true, "Always create a superinitial state");

int fstreverse_main(int argc, char **argv);

int main(int argc, char **argv) { return fstreverse_main(argc, argv); }
