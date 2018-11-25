// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <istream>
#include <string>

#include <fst/script/compile.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void CompileFst(std::istream &istrm, const string &source, const string &dest,
                const string &fst_type, const string &arc_type,
                const SymbolTable *isyms, const SymbolTable *osyms,
                const SymbolTable *ssyms, bool accep, bool ikeep, bool okeep,
                bool nkeep, bool allow_negative_labels) {
  std::unique_ptr<FstClass> fst(
      CompileFstInternal(istrm, source, fst_type, arc_type, isyms, osyms, ssyms,
                         accep, ikeep, okeep, nkeep, allow_negative_labels));
  fst->Write(dest);
}

FstClass *CompileFstInternal(std::istream &istrm, const string &source,
                             const string &fst_type, const string &arc_type,
                             const SymbolTable *isyms, const SymbolTable *osyms,
                             const SymbolTable *ssyms, bool accep, bool ikeep,
                             bool okeep, bool nkeep,
                             bool allow_negative_labels) {
  CompileFstInnerArgs iargs(istrm, source, fst_type, isyms, osyms, ssyms, accep,
                            ikeep, okeep, nkeep, allow_negative_labels);
  CompileFstArgs args(iargs);
  Apply<Operation<CompileFstArgs>>("CompileFstInternal", arc_type, &args);
  return args.retval;
}

// This registers 2; 1 does not require registration.
REGISTER_FST_OPERATION(CompileFstInternal, StdArc, CompileFstArgs);
REGISTER_FST_OPERATION(CompileFstInternal, LogArc, CompileFstArgs);
REGISTER_FST_OPERATION(CompileFstInternal, Log64Arc, CompileFstArgs);

}  // namespace script
}  // namespace fst
