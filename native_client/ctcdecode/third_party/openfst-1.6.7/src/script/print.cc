// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <ostream>
#include <string>

#include <fst/script/fst-class.h>
#include <fst/script/print.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void PrintFst(const FstClass &fst, std::ostream &ostrm, const string &dest,
              const SymbolTable *isyms, const SymbolTable *osyms,
              const SymbolTable *ssyms, bool accept, bool show_weight_one,
              const string &missing_sym) {
  const auto sep = FLAGS_fst_field_separator.substr(0, 1);
  FstPrinterArgs args(fst, isyms, osyms, ssyms, accept, show_weight_one, &ostrm,
                      dest, sep, missing_sym);
  Apply<Operation<FstPrinterArgs>>("PrintFst", fst.ArcType(), &args);
}

REGISTER_FST_OPERATION(PrintFst, StdArc, FstPrinterArgs);
REGISTER_FST_OPERATION(PrintFst, LogArc, FstPrinterArgs);
REGISTER_FST_OPERATION(PrintFst, Log64Arc, FstPrinterArgs);

}  // namespace script
}  // namespace fst
