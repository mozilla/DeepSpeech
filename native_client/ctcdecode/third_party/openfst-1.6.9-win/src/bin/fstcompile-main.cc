// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Creates binary FSTs from simple text format used by AT&T.

#include <cstring>

#include <fstream>
#include <istream>
#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/compile.h>

DECLARE_bool(acceptor);
DECLARE_string(arc_type);
DECLARE_string(fst_type);
DECLARE_string(isymbols);
DECLARE_string(osymbols);
DECLARE_string(ssymbols);
DECLARE_bool(keep_isymbols);
DECLARE_bool(keep_osymbols);
DECLARE_bool(keep_state_numbering);
DECLARE_bool(allow_negative_labels);

int fstcompile_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::SymbolTable;
  using fst::SymbolTableTextOptions;

  string usage = "Creates binary FSTs from simple text format.\n\n  Usage: ";
  usage += argv[0];
  usage += " [text.fst [binary.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  string source = "standard input";
  std::ifstream fstrm;
  if (argc > 1 && strcmp(argv[1], "-") != 0) {
    fstrm.open(argv[1]);
    if (!fstrm) {
      LOG(ERROR) << argv[0] << ": Open failed, file = " << argv[1];
      return 1;
    }
    source = argv[1];
  }
  std::istream &istrm = fstrm.is_open() ? fstrm : std::cin;

  const SymbolTableTextOptions opts(FLAGS_allow_negative_labels);

  std::unique_ptr<const SymbolTable> isyms;
  if (!FLAGS_isymbols.empty()) {
    isyms.reset(SymbolTable::ReadText(FLAGS_isymbols, opts));
    if (!isyms) return 1;
  }

  std::unique_ptr<const SymbolTable> osyms;
  if (!FLAGS_osymbols.empty()) {
    osyms.reset(SymbolTable::ReadText(FLAGS_osymbols, opts));
    if (!osyms) return 1;
  }

  std::unique_ptr<const SymbolTable> ssyms;
  if (!FLAGS_ssymbols.empty()) {
    ssyms.reset(SymbolTable::ReadText(FLAGS_ssymbols));
    if (!ssyms) return 1;
  }

  const string dest = argc > 2 ? argv[2] : "";

  s::CompileFst(istrm, source, dest, FLAGS_fst_type, FLAGS_arc_type,
                isyms.get(), osyms.get(), ssyms.get(), FLAGS_acceptor,
                FLAGS_keep_isymbols, FLAGS_keep_osymbols,
                FLAGS_keep_state_numbering, FLAGS_allow_negative_labels);

  return 0;
}
