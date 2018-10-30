// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_COMPILE_H_
#define FST_SCRIPT_COMPILE_H_

#include <istream>
#include <memory>

#include <fst/script/arg-packs.h>
#include <fst/script/compile-impl.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

// This operation exists in two forms. 1 is a void operation which writes the
// compiled machine to disk; 2 returns an FstClass. I/O should normally be done
// using the binary format for efficiency, so users are STRONGLY ENCOURAGED to
// use 1 or to construct FSTs using the C++ FST mutation operations.

// Note: it is safe to pass these strings as references because
// this struct is only used to pass them deeper in the call graph.
// Be sure you understand why this is so before using this struct
// for anything else!
struct CompileFstInnerArgs {
  std::istream &istrm;
  const string &source;
  const string &fst_type;
  const fst::SymbolTable *isyms;
  const fst::SymbolTable *osyms;
  const fst::SymbolTable *ssyms;
  const bool accep;
  const bool ikeep;
  const bool okeep;
  const bool nkeep;
  const bool allow_negative_labels;

  CompileFstInnerArgs(std::istream &istrm, const string &source,
                      const string &fst_type, const fst::SymbolTable *isyms,
                      const fst::SymbolTable *osyms,
                      const fst::SymbolTable *ssyms, bool accep, bool ikeep,
                      bool okeep, bool nkeep,
                      bool allow_negative_labels = false)
      : istrm(istrm),
        source(source),
        fst_type(fst_type),
        isyms(isyms),
        osyms(osyms),
        ssyms(ssyms),
        accep(accep),
        ikeep(ikeep),
        okeep(okeep),
        nkeep(nkeep),
        allow_negative_labels(allow_negative_labels) {}
};

using CompileFstArgs = WithReturnValue<FstClass *, CompileFstInnerArgs>;

template <class Arc>
void CompileFstInternal(CompileFstArgs *args) {
  using fst::Convert;
  using fst::Fst;
  using fst::FstCompiler;
  FstCompiler<Arc> fstcompiler(
      args->args.istrm, args->args.source, args->args.isyms, args->args.osyms,
      args->args.ssyms, args->args.accep, args->args.ikeep, args->args.okeep,
      args->args.nkeep, args->args.allow_negative_labels);
  const Fst<Arc> *fst = &fstcompiler.Fst();
  std::unique_ptr<const Fst<Arc>> owned_fst;
  if (args->args.fst_type != "vector") {
    owned_fst.reset(Convert<Arc>(*fst, args->args.fst_type));
    if (!owned_fst) {
      FSTERROR() << "Failed to convert FST to desired type: "
                 << args->args.fst_type;
    }
    fst = owned_fst.get();
  }
  args->retval = fst ? new FstClass(*fst) : nullptr;
}

void CompileFst(std::istream &istrm, const string &source, const string &dest,
                const string &fst_type, const string &arc_type,
                const SymbolTable *isyms, const SymbolTable *osyms,
                const SymbolTable *ssyms, bool accep, bool ikeep, bool okeep,
                bool nkeep, bool allow_negative_labels);

FstClass *CompileFstInternal(std::istream &istrm, const string &source,
                             const string &fst_type, const string &arc_type,
                             const SymbolTable *isyms, const SymbolTable *osyms,
                             const SymbolTable *ssyms, bool accep, bool ikeep,
                             bool okeep, bool nkeep,
                             bool allow_negative_labels);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_COMPILE_H_
