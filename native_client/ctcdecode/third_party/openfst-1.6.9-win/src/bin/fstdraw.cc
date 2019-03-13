// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(acceptor, false, "Input in acceptor format");
DEFINE_string(isymbols, "", "Input label symbol table");
DEFINE_string(osymbols, "", "Output label symbol table");
DEFINE_string(ssymbols, "", "State label symbol table");
DEFINE_bool(numeric, false, "Print numeric labels");
DEFINE_int32(precision, 5, "Set precision (number of char/float)");
DEFINE_string(float_format, "g",
              "Floating-point format, one of: \"e\", \"f\", or \"g\"");
DEFINE_bool(show_weight_one, false,
            "Print/draw arc weights and final weights equal to Weight::One()");
DEFINE_string(title, "", "Set figure title");
DEFINE_bool(portrait, false, "Portrait mode (def: landscape)");
DEFINE_bool(vertical, false, "Draw bottom-to-top instead of left-to-right");
DEFINE_int32(fontsize, 14, "Set fontsize");
DEFINE_double(height, 11, "Set height");
DEFINE_double(width, 8.5, "Set width");
DEFINE_double(nodesep, 0.25,
              "Set minimum separation between nodes (see dot documentation)");
DEFINE_double(ranksep, 0.40,
              "Set minimum separation between ranks (see dot documentation)");
DEFINE_bool(allow_negative_labels, false,
            "Allow negative labels (not recommended; may cause conflicts)");

int fstdraw_main(int argc, char **argv);

int main(int argc, char **argv) { return fstdraw_main(argc, argv); }
