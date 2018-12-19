// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_string(fst_type, "vector", "Output FST type");

int fstconvert_main(int argc, char **argv);

int main(int argc, char **argv) { return fstconvert_main(argc, argv); }
