// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(closure_plus, false,
            "Do not add the empty path (T+ instead of T*)?");

int fstclosure_main(int argc, char **argv);

int main(int argc, char **argv) { return fstclosure_main(argc, argv); }
