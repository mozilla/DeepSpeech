// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/linear/linearscript.h>

#include <fst/flags.h>

DEFINE_string(arc_type, "standard", "Output arc type");

DEFINE_string(epsilon_symbol, "<eps>", "Epsilon symbol");
DEFINE_string(unknown_symbol, "<unk>", "Unknown word symbol");

DEFINE_string(vocab, "", "Path to the vocabulary file");
DEFINE_string(out, "", "Path to the output binary");

DEFINE_string(save_isymbols, "", "Save input symbol table to file");
DEFINE_string(save_fsymbols, "", "Save feature symbol table to file");
DEFINE_string(save_osymbols, "", "Save output symbol table to file");

int main(int argc, char **argv) {
  // TODO(wuke): more detailed usage
  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(argv[0], &argc, &argv, true);
  fst::script::ValidateDelimiter();
  fst::script::ValidateEmptySymbol();

  if (argc == 1) {
    ShowUsage();
    return 1;
  }

  fst::script::LinearCompile(FLAGS_arc_type, FLAGS_epsilon_symbol,
                                 FLAGS_unknown_symbol, FLAGS_vocab, argv + 1,
                                 argc - 1, FLAGS_out, FLAGS_save_isymbols,
                                 FLAGS_save_fsymbols, FLAGS_save_osymbols);
}
