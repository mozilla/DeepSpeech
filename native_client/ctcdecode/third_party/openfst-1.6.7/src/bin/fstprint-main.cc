// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Prints out binary FSTs in simple text format used by AT&T.

#include <cstring>

#include <fstream>
#include <memory>
#include <ostream>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/print.h>

DECLARE_bool(acceptor);
DECLARE_string(isymbols);
DECLARE_string(osymbols);
DECLARE_string(ssymbols);
DECLARE_bool(numeric);
DECLARE_string(save_isymbols);
DECLARE_string(save_osymbols);
DECLARE_bool(show_weight_one);
DECLARE_bool(allow_negative_labels);
DECLARE_string(missing_symbol);

int fstprint_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::SymbolTable;
  using fst::SymbolTableTextOptions;

  string usage = "Prints out binary FSTs in simple text format.\n\n  Usage: ";
  usage += argv[0];
  usage += " [binary.fst [text.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> fst(FstClass::Read(in_name));
  if (!fst) return 1;

  string dest = "standard output";
  std::ofstream fstrm;
  if (argc == 3) {
    fstrm.open(argv[2]);
    if (!fstrm) {
      LOG(ERROR) << argv[0] << ": Open failed, file = " << argv[2];
      return 1;
    }
    dest = argv[2];
  }
  std::ostream &ostrm = fstrm.is_open() ? fstrm : std::cout;
  ostrm.precision(9);

  const SymbolTableTextOptions opts(FLAGS_allow_negative_labels);

  std::unique_ptr<const SymbolTable> isyms;
  if (!FLAGS_isymbols.empty() && !FLAGS_numeric) {
    isyms.reset(SymbolTable::ReadText(FLAGS_isymbols, opts));
    if (!isyms) return 1;
  }

  std::unique_ptr<const SymbolTable> osyms;
  if (!FLAGS_osymbols.empty() && !FLAGS_numeric) {
    osyms.reset(SymbolTable::ReadText(FLAGS_osymbols, opts));
    if (!osyms) return 1;
  }

  std::unique_ptr<const SymbolTable> ssyms;
  if (!FLAGS_ssymbols.empty() && !FLAGS_numeric) {
    ssyms.reset(SymbolTable::ReadText(FLAGS_ssymbols));
    if (!ssyms) return 1;
  }

  if (!isyms && !FLAGS_numeric && fst->InputSymbols()) {
    isyms.reset(fst->InputSymbols()->Copy());
  }

  if (!osyms && !FLAGS_numeric && fst->OutputSymbols()) {
    osyms.reset(fst->OutputSymbols()->Copy());
  }

  s::PrintFst(*fst, ostrm, dest, isyms.get(), osyms.get(), ssyms.get(),
              FLAGS_acceptor, FLAGS_show_weight_one, FLAGS_missing_symbol);

  if (isyms && !FLAGS_save_isymbols.empty()) {
    if (!isyms->WriteText(FLAGS_save_isymbols)) return 1;
  }

  if (osyms && !FLAGS_save_osymbols.empty()) {
    if (!osyms->WriteText(FLAGS_save_osymbols)) return 1;
  }

  return 0;
}
