// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_string(compose_filter, "auto",
              "Composition filter, one of: \"alt_sequence\", \"auto\", "
              "\"match\", \"null\", \"sequence\", \"trivial\"");
DEFINE_bool(connect, true, "Trim output");

int fstdifference_main(int argc, char **argv);

int main(int argc, char **argv) { return fstdifference_main(argc, argv); }
