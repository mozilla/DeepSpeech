// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_string(isymbols, "", "Input label symbol table");
DEFINE_string(osymbols, "", "Output label symbol table");
DEFINE_bool(clear_isymbols, false, "Clear input symbol table");
DEFINE_bool(clear_osymbols, false, "Clear output symbol table");
DEFINE_string(relabel_ipairs, "", "Input relabel pairs (numeric)");
DEFINE_string(relabel_opairs, "", "Output relabel pairs (numeric)");
DEFINE_string(save_isymbols, "", "Save fst file's input symbol table to file");
DEFINE_string(save_osymbols, "", "Save fst file's output symbol table to file");
DEFINE_bool(allow_negative_labels, false,
            "Allow negative labels (not recommended; may cause conflicts)");
DEFINE_bool(verify, false, "Verify fst properities before saving");

int fstsymbols_main(int argc, char **argv);

int main(int argc, char **argv) { return fstsymbols_main(argc, argv); }
