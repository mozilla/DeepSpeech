// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Relabels input or output space of an FST.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/util.h>
#include <fst/script/relabel.h>
#include <fst/script/weight-class.h>

DECLARE_string(isymbols);
DECLARE_string(osymbols);
DECLARE_string(relabel_isymbols);
DECLARE_string(relabel_osymbols);
DECLARE_string(relabel_ipairs);
DECLARE_string(relabel_opairs);
DECLARE_string(unknown_isymbol);
DECLARE_string(unknown_osymbol);
DECLARE_bool(allow_negative_labels);

int fstrelabel_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::MutableFstClass;
  using fst::SymbolTable;
  using fst::SymbolTableTextOptions;

  string usage =
      "Relabels the input and/or the output labels of the FST.\n\n"
      "  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fst]]\n";
  usage += "\n Using SymbolTables flags:\n";
  usage += "  --relabel_isymbols isyms.map\n";
  usage += "  --relabel_osymbols osyms.map\n";
  usage += "\n Using numeric labels flags:\n";
  usage += "  --relabel_ipairs ipairs.txt\n";
  usage += "  --relabel_opairs opairs.txt\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
  if (!fst) return 1;

  // Relabel with symbol tables.
  const SymbolTableTextOptions opts(FLAGS_allow_negative_labels);

  if (!FLAGS_relabel_isymbols.empty() || !FLAGS_relabel_osymbols.empty()) {
    bool attach_new_isymbols = (fst->InputSymbols() != nullptr);
    std::unique_ptr<const SymbolTable> old_isymbols(
        FLAGS_isymbols.empty() ? nullptr
                               : SymbolTable::ReadText(FLAGS_isymbols, opts));
    const std::unique_ptr<const SymbolTable> relabel_isymbols(
        FLAGS_relabel_isymbols.empty()
            ? nullptr
            : SymbolTable::ReadText(FLAGS_relabel_isymbols, opts));
    bool attach_new_osymbols = (fst->OutputSymbols() != nullptr);
    std::unique_ptr<const SymbolTable> old_osymbols(
        FLAGS_osymbols.empty() ? nullptr
                               : SymbolTable::ReadText(FLAGS_osymbols, opts));
    const std::unique_ptr<const SymbolTable> relabel_osymbols(
        FLAGS_relabel_osymbols.empty()
            ? nullptr
            : SymbolTable::ReadText(FLAGS_relabel_osymbols, opts));
    s::Relabel(fst.get(),
               old_isymbols ? old_isymbols.get() : fst->InputSymbols(),
               relabel_isymbols.get(), FLAGS_unknown_isymbol,
               attach_new_isymbols,
               old_osymbols ? old_osymbols.get() : fst->OutputSymbols(),
               relabel_osymbols.get(), FLAGS_unknown_osymbol,
               attach_new_osymbols);
  } else {
    // Reads in relabeling pairs.
    std::vector<s::LabelPair> ipairs;
    std::vector<s::LabelPair> opairs;
    if (!FLAGS_relabel_ipairs.empty()) {
      if (!fst::ReadLabelPairs(FLAGS_relabel_ipairs, &ipairs,
                                   FLAGS_allow_negative_labels))
        return 1;
    }
    if (!FLAGS_relabel_opairs.empty()) {
      if (!fst::ReadLabelPairs(FLAGS_relabel_opairs, &opairs,
                                   FLAGS_allow_negative_labels))
        return 1;
    }
    s::Relabel(fst.get(), ipairs, opairs);
  }

  return !fst->Write(out_name);
}
