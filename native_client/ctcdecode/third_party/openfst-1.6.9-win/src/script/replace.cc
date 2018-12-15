// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/replace.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Replace(const std::vector<LabelFstClassPair> &pairs, MutableFstClass *ofst,
             const ReplaceOptions &opts) {
  if (!pairs.empty()) {
    for (auto it = pairs.begin(); it != pairs.end() - 1; ++it) {
      if (!internal::ArcTypesMatch(*it->second, *(it + 1)->second, "Replace")) {
        ofst->SetProperties(kError, kError);
        return;
      }
    }
    if (!internal::ArcTypesMatch(*pairs[0].second, *ofst, "Replace")) {
      ofst->SetProperties(kError, kError);
      return;
    }
  }
  ReplaceArgs args(pairs, ofst, opts);
  Apply<Operation<ReplaceArgs>>("Replace", ofst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Replace, StdArc, ReplaceArgs);
REGISTER_FST_OPERATION(Replace, LogArc, ReplaceArgs);
REGISTER_FST_OPERATION(Replace, Log64Arc, ReplaceArgs);

}  // namespace script
}  // namespace fst
