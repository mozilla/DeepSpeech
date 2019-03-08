// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(encode_labels, false, "Encode output labels");
DEFINE_bool(encode_weights, false, "Encode weights");
DEFINE_bool(encode_reuse, false, "Re-use existing codex");
DEFINE_bool(decode, false, "Decode labels and/or weights");

int fstencode_main(int argc, char **argv);

int main(int argc, char **argv) { return fstencode_main(argc, argv); }
