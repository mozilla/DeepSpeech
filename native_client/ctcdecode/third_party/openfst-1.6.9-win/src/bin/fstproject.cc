// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(project_output, false, "Project on output (vs. input)");

int fstproject_main(int argc, char **argv);

int main(int argc, char **argv) { return fstproject_main(argc, argv); }
