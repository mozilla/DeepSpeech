// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/flags.h>

DEFINE_string(isymbols, "", "Input label symbol table");
DEFINE_string(osymbols, "", "Output label symbol table");
DEFINE_string(relabel_isymbols, "", "Input symbol set to relabel to");
DEFINE_string(relabel_osymbols, "", "Output symbol set to relabel to");
DEFINE_string(relabel_ipairs, "", "Input relabel pairs (numeric)");
DEFINE_string(relabel_opairs, "", "Output relabel pairs (numeric)");
DEFINE_string(unknown_isymbol, "",
    "Input symbol to use to relabel OOVs (default: OOVs are errors)");
DEFINE_string(unknown_osymbol, "",
    "Output symbol to use to relabel OOVs (default: OOVs are errors)");
DEFINE_bool(allow_negative_labels, false,
    "Allow negative labels (not recommended; may cause conflicts)");

int fstrelabel_main(int argc, char **argv);

int main(int argc, char **argv) { return fstrelabel_main(argc, argv); }
