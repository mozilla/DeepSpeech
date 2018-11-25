// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Convenience file for including all MPDT operations at once, and/or
// registering them for new arc types.

#ifndef FST_EXTENSIONS_MPDT_MPDTSCRIPT_H_
#define FST_EXTENSIONS_MPDT_MPDTSCRIPT_H_

#include <algorithm>
#include <utility>
#include <vector>

#include <fst/log.h>
#include <fst/compose.h>  // for ComposeOptions
#include <fst/util.h>

#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/shortest-path.h>

#include <fst/extensions/mpdt/compose.h>
#include <fst/extensions/mpdt/expand.h>
#include <fst/extensions/mpdt/info.h>
#include <fst/extensions/mpdt/reverse.h>

#include <fst/extensions/pdt/pdtscript.h>  // For LabelClassPair,
                                               // FstClassPair, and to detect
                                               // any collisions.

namespace fst {
namespace script {

using MPdtComposeArgs =
    std::tuple<const FstClass &, const FstClass &,
               const std::vector<LabelPair> &, const std::vector<int64> &,
               MutableFstClass *, const MPdtComposeOptions &, bool>;

template <class Arc>
void MPdtCompose(MPdtComposeArgs *args) {
  const Fst<Arc> &ifst1 = *(std::get<0>(*args).GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(std::get<1>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<4>(*args)->GetMutableFst<Arc>();
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<2>(*args).size());
  std::copy(std::get<2>(*args).begin(), std::get<2>(*args).end(),
            typed_parens.begin());
  using Level = typename Arc::Label;
  std::vector<Level> typed_assignments(std::get<3>(*args).size());
  std::copy(std::get<3>(*args).begin(), std::get<3>(*args).end(),
            typed_assignments.begin());
  if (std::get<6>(*args)) {
    Compose(ifst1, typed_parens, typed_assignments, ifst2, ofst,
            std::get<5>(*args));
  } else {
    Compose(ifst1, ifst2, typed_parens, typed_assignments, ofst,
            std::get<5>(*args));
  }
}

void MPdtCompose(const FstClass &ifst1, const FstClass &ifst2,
                 const std::vector<LabelPair> &parens,
                 const std::vector<int64> &assignments, MutableFstClass *ofst,
                 const MPdtComposeOptions &copts, bool left_pdt);

using MPdtExpandArgs =
    std::tuple<const FstClass &, const std::vector<LabelPair> &,
               const std::vector<int64> &, MutableFstClass *,
               const MPdtExpandOptions &>;

template <class Arc>
void MPdtExpand(MPdtExpandArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<3>(*args)->GetMutableFst<Arc>();
  // In case Arc::Label is not the same as FstClass::Label, we make copies.
  // Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  using Level = typename Arc::Label;
  std::vector<Level> typed_assignments(std::get<2>(*args).size());
  std::copy(std::get<2>(*args).begin(), std::get<2>(*args).end(),
            typed_assignments.begin());
  Expand(fst, typed_parens, typed_assignments, ofst,
         MPdtExpandOptions(std::get<4>(*args).connect,
                           std::get<4>(*args).keep_parentheses));
}

void MPdtExpand(const FstClass &ifst, const std::vector<LabelPair> &parens,
                const std::vector<int64> &assignments, MutableFstClass *ofst,
                const MPdtExpandOptions &opts);

using MPdtReverseArgs =
    std::tuple<const FstClass &, const std::vector<LabelPair> &,
               std::vector<int64> *, MutableFstClass *>;

template <class Arc>
void MPdtReverse(MPdtReverseArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<3>(*args)->GetMutableFst<Arc>();
  // In case Arc::Label is not the same as FstClass::Label, we make copies.
  // Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  using Level = typename Arc::Label;
  std::vector<Level> typed_assignments(std::get<2>(*args)->size());
  std::copy(std::get<2>(*args)->begin(), std::get<2>(*args)->end(),
            typed_assignments.begin());
  Reverse(fst, typed_parens, &typed_assignments, ofst);
  // Reassign stack assignments to input assignment vector.
  std::copy(typed_assignments.begin(), typed_assignments.end(),
            std::get<2>(*args)->begin());
}

void MPdtReverse(const FstClass &ifst, const std::vector<LabelPair> &parens,
                 std::vector<int64> *assignments, MutableFstClass *ofst);

using PrintMPdtInfoArgs =
    std::tuple<const FstClass &, const std::vector<LabelPair> &,
               const std::vector<int64> &>;

template <class Arc>
void PrintMPdtInfo(PrintMPdtInfoArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  // In case Arc::Label is not the same as FstClass::Label, we make copies.
  // Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  using Level = typename Arc::Label;
  std::vector<Level> typed_assignments(std::get<2>(*args).size());
  std::copy(std::get<2>(*args).begin(), std::get<2>(*args).end(),
            typed_assignments.begin());
  MPdtInfo<Arc> mpdtinfo(fst, typed_parens, typed_assignments);
  mpdtinfo.Print();
}

void PrintMPdtInfo(const FstClass &ifst, const std::vector<LabelPair> &parens,
                   const std::vector<int64> &assignments);

}  // namespace script
}  // namespace fst

#define REGISTER_FST_MPDT_OPERATIONS(ArcType)                    \
  REGISTER_FST_OPERATION(MPdtCompose, ArcType, MPdtComposeArgs); \
  REGISTER_FST_OPERATION(MPdtExpand, ArcType, MPdtExpandArgs);   \
  REGISTER_FST_OPERATION(MPdtReverse, ArcType, MPdtReverseArgs); \
  REGISTER_FST_OPERATION(PrintMPdtInfo, ArcType, PrintMPdtInfoArgs)
#endif  // FST_EXTENSIONS_MPDT_MPDTSCRIPT_H_
