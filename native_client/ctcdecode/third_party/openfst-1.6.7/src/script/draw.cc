// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <ostream>
#include <string>

#include <fst/script/draw.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void DrawFst(const FstClass &fst, const SymbolTable *isyms,
             const SymbolTable *osyms, const SymbolTable *ssyms, bool accep,
             const string &title, float width, float height, bool portrait,
             bool vertical, float ranksep, float nodesep, int fontsize,
             int precision, const string &float_format, bool show_weight_one,
             std::ostream *ostrm, const string &dest) {
  FstDrawerArgs args(fst, isyms, osyms, ssyms, accep, title, width, height,
                     portrait, vertical, ranksep, nodesep, fontsize, precision,
                     float_format, show_weight_one, ostrm, dest);
  Apply<Operation<FstDrawerArgs>>("DrawFst", fst.ArcType(), &args);
}

REGISTER_FST_OPERATION(DrawFst, StdArc, FstDrawerArgs);
REGISTER_FST_OPERATION(DrawFst, LogArc, FstDrawerArgs);
REGISTER_FST_OPERATION(DrawFst, Log64Arc, FstDrawerArgs);

}  // namespace script
}  // namespace fst
