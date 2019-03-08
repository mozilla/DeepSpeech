// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_REPLACE_H_
#define FST_SCRIPT_REPLACE_H_

#include <tuple>
#include <utility>
#include <vector>

#include <fst/replace.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

struct ReplaceOptions {
  const int64_t root;                          // Root rule for expansion.
  const ReplaceLabelType call_label_type;    // How to label call arc.
  const ReplaceLabelType return_label_type;  // How to label return arc.
  const int64_t return_label;                  // Specifies return arc label.

  explicit ReplaceOptions(int64_t root,
      ReplaceLabelType call_label_type = REPLACE_LABEL_INPUT,
      ReplaceLabelType return_label_type = REPLACE_LABEL_NEITHER,
      int64_t return_label = 0)
      : root(root),
        call_label_type(call_label_type),
        return_label_type(return_label_type),
        return_label(return_label) {}
};

using LabelFstClassPair = std::pair<int64_t, const FstClass *>;

using ReplaceArgs = std::tuple<const std::vector<LabelFstClassPair> &,
                               MutableFstClass *, const ReplaceOptions &>;

template <class Arc>
void Replace(ReplaceArgs *args) {
  using LabelFstPair = std::pair<typename Arc::Label, const Fst<Arc> *>;
  // Now that we know the arc type, we construct a vector of
  // std::pair<real label, real fst> that the real Replace will use.
  const auto &untyped_pairs = std::get<0>(*args);
  std::vector<LabelFstPair> typed_pairs;
  typed_pairs.reserve(untyped_pairs.size());
  for (const auto &untyped_pair : untyped_pairs) {
    typed_pairs.emplace_back(untyped_pair.first,  // Converts label.
                             untyped_pair.second->GetFst<Arc>());
  }
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  const auto &opts = std::get<2>(*args);
  ReplaceFstOptions<Arc> typed_opts(opts.root, opts.call_label_type,
                                    opts.return_label_type, opts.return_label);
  ReplaceFst<Arc> rfst(typed_pairs, typed_opts);
  // Checks for cyclic dependencies before attempting expansion.
  if (rfst.CyclicDependencies()) {
    FSTERROR() << "Replace: Cyclic dependencies detected; cannot expand";
    ofst->SetProperties(kError, kError);
    return;
  }
  typed_opts.gc = true;     // Caching options to speed up batch copy.
  typed_opts.gc_limit = 0;
  *ofst = rfst;
}

void Replace(const std::vector<LabelFstClassPair> &pairs,
             MutableFstClass *ofst, const ReplaceOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_REPLACE_H_
