// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_bool(acceptor, false, "Input in acceptor format?");
DEFINE_string(isymbols, "", "Input label symbol table");
DEFINE_string(osymbols, "", "Output label symbol table");
DEFINE_string(ssymbols, "", "State label symbol table");
DEFINE_bool(numeric, false, "Print numeric labels?");
DEFINE_string(save_isymbols, "", "Save input symbol table to file");
DEFINE_string(save_osymbols, "", "Save output symbol table to file");
DEFINE_bool(show_weight_one, false,
            "Print/draw arc weights and final weights equal to semiring One?");
DEFINE_bool(allow_negative_labels, false,
            "Allow negative labels (not recommended; may cause conflicts)?");
DEFINE_string(missing_symbol, "",
              "Symbol to print when lookup fails (default raises error)");

int fstprint_main(int argc, char **argv);

int main(int argc, char **argv) { return fstprint_main(argc, argv); }
