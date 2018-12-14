// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PRINT_H_
#define FST_SCRIPT_PRINT_H_

#include <ostream>

#include <fst/flags.h>
#include <fst/script/fst-class.h>
#include <fst/script/print-impl.h>

DECLARE_string(fst_field_separator);

namespace fst {
namespace script {

// Note: it is safe to pass these strings as references because
// this struct is only used to pass them deeper in the call graph.
// Be sure you understand why this is so before using this struct
// for anything else!
struct FstPrinterArgs {
  const FstClass &fst;
  const SymbolTable *isyms;
  const SymbolTable *osyms;
  const SymbolTable *ssyms;
  const bool accept;
  const bool show_weight_one;
  std::ostream *ostrm;
  const string &dest;
  const string &sep;  // NOLINT
  const string &missing_symbol;

  FstPrinterArgs(const FstClass &fst, const SymbolTable *isyms,
                 const SymbolTable *osyms, const SymbolTable *ssyms,
                 bool accept, bool show_weight_one, std::ostream *ostrm,
                 const string &dest, const string &sep,
                 const string &missing_sym = "")
      : fst(fst),
        isyms(isyms),
        osyms(osyms),
        ssyms(ssyms),
        accept(accept),
        show_weight_one(show_weight_one),
        ostrm(ostrm),
        dest(dest),
        sep(sep),
        missing_symbol(missing_sym) {}
};

template <class Arc>
void PrintFst(FstPrinterArgs *args) {
  const Fst<Arc> &fst = *(args->fst.GetFst<Arc>());
  FstPrinter<Arc> fstprinter(fst, args->isyms, args->osyms, args->ssyms,
                             args->accept, args->show_weight_one, args->sep,
                             args->missing_symbol);
  fstprinter.Print(args->ostrm, args->dest);
}

void PrintFst(const FstClass &fst, std::ostream &ostrm, const string &dest,
              const SymbolTable *isyms, const SymbolTable *osyms,
              const SymbolTable *ssyms, bool accept, bool show_weight_one,
              const string &missing_sym = "");

// The same, but with more sensible defaults.
template <class Arc>
void PrintFst(const Fst<Arc> &fst, std::ostream &ostrm, const string &dest = "",
              const SymbolTable *isyms = nullptr,
              const SymbolTable *osyms = nullptr,
              const SymbolTable *ssyms = nullptr) {
  const string sep = FLAGS_fst_field_separator.substr(0, 1);
  FstPrinter<Arc> fstprinter(fst, isyms, osyms, ssyms, true, true, sep);
  fstprinter.Print(&ostrm, dest);
}

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PRINT_H_
