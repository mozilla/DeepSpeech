// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Convenience file for including all PDT operations at once, and/or
// registering them for new arc types.

#ifndef FST_EXTENSIONS_PDT_PDTSCRIPT_H_
#define FST_EXTENSIONS_PDT_PDTSCRIPT_H_

#include <algorithm>
#include <utility>
#include <vector>

#include <fst/log.h>
#include <fst/compose.h>  // for ComposeOptions
#include <fst/util.h>

#include <fst/script/arg-packs.h>
#include <fst/script/fstscript.h>
#include <fst/script/shortest-path.h>

#include <fst/extensions/pdt/compose.h>
#include <fst/extensions/pdt/expand.h>
#include <fst/extensions/pdt/info.h>
#include <fst/extensions/pdt/replace.h>
#include <fst/extensions/pdt/reverse.h>
#include <fst/extensions/pdt/shortest-path.h>

namespace fst {
namespace script {

using PdtComposeArgs =
    std::tuple<const FstClass &, const FstClass &,
               const std::vector<LabelPair> &, MutableFstClass *,
               const PdtComposeOptions &, bool>;

template <class Arc>
void PdtCompose(PdtComposeArgs *args) {
  const Fst<Arc> &ifst1 = *(std::get<0>(*args).GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(std::get<1>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<3>(*args)->GetMutableFst<Arc>();
  // In case Arc::Label is not the same as FstClass::Label, we make a
  // copy. Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<2>(*args).size());
  std::copy(std::get<2>(*args).begin(), std::get<2>(*args).end(),
            typed_parens.begin());
  if (std::get<5>(*args)) {
    Compose(ifst1, typed_parens, ifst2, ofst, std::get<4>(*args));
  } else {
    Compose(ifst1, ifst2, typed_parens, ofst, std::get<4>(*args));
  }
}

void PdtCompose(const FstClass &ifst1, const FstClass &ifst2,
                const std::vector<LabelPair> &parens,
                MutableFstClass *ofst, const PdtComposeOptions &opts,
                bool left_pdt);

struct PdtExpandOptions {
  bool connect;
  bool keep_parentheses;
  const WeightClass &weight_threshold;

  PdtExpandOptions(bool c, bool k, const WeightClass &w)
      : connect(c), keep_parentheses(k), weight_threshold(w) {}
};

using PdtExpandArgs =
    std::tuple<const FstClass &, const std::vector<LabelPair> &,
               MutableFstClass *, const PdtExpandOptions &>;

template <class Arc>
void PdtExpand(PdtExpandArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<2>(*args)->GetMutableFst<Arc>();
  // In case Arc::Label is not the same as FstClass::Label, we make a
  // copy. Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  Expand(fst, typed_parens, ofst,
         fst::PdtExpandOptions<Arc>(
             std::get<3>(*args).connect, std::get<3>(*args).keep_parentheses,
             *(std::get<3>(*args)
                   .weight_threshold.GetWeight<typename Arc::Weight>())));
}

void PdtExpand(const FstClass &ifst, const std::vector<LabelPair> &parens,
               MutableFstClass *ofst, const PdtExpandOptions &opts);

void PdtExpand(const FstClass &ifst, const std::vector<LabelPair> &parens,
               MutableFstClass *ofst, bool connect, bool keep_parentheses,
               const WeightClass &weight_threshold);

using PdtReplaceArgs =
    std::tuple<const std::vector<LabelFstClassPair> &, MutableFstClass *,
               std::vector<LabelPair> *, int64_t, PdtParserType, int64_t,
               const string &, const string &>;

template <class Arc>
void PdtReplace(PdtReplaceArgs *args) {
  const auto &untyped_pairs = std::get<0>(*args);
  auto size = untyped_pairs.size();
  std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>> typed_pairs(
      size);
  for (size_t i = 0; i < size; ++i) {
    typed_pairs[i].first = untyped_pairs[i].first;
    typed_pairs[i].second = untyped_pairs[i].second->GetFst<Arc>();
  }
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens;
  const PdtReplaceOptions<Arc> opts(std::get<3>(*args), std::get<4>(*args),
                                    std::get<5>(*args), std::get<6>(*args),
                                    std::get<7>(*args));
  Replace(typed_pairs, ofst, &typed_parens, opts);
  // Copies typed parens into arg3.
  std::get<2>(*args)->resize(typed_parens.size());
  std::copy(typed_parens.begin(), typed_parens.end(),
            std::get<2>(*args)->begin());
}

void PdtReplace(const std::vector<LabelFstClassPair> &pairs,
                MutableFstClass *ofst, std::vector<LabelPair> *parens,
                int64_t root, PdtParserType parser_type = PDT_LEFT_PARSER,
                int64_t start_paren_labels = kNoLabel,
                const string &left_paren_prefix = "(_",
                const string &right_paren_prefix = "_)");

using PdtReverseArgs =
    std::tuple<const FstClass &, const std::vector<LabelPair> &,
               MutableFstClass *>;

template <class Arc>
void PdtReverse(PdtReverseArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<2>(*args)->GetMutableFst<Arc>();
  // In case Arc::Label is not the same as FstClass::Label, we make a
  // copy. Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  Reverse(fst, typed_parens, ofst);
}

void PdtReverse(const FstClass &ifst, const std::vector<LabelPair> &,
                MutableFstClass *ofst);

// PDT SHORTESTPATH

struct PdtShortestPathOptions {
  QueueType queue_type;
  bool keep_parentheses;
  bool path_gc;

  PdtShortestPathOptions(QueueType qt = FIFO_QUEUE, bool kp = false,
                         bool gc = true)
      : queue_type(qt), keep_parentheses(kp), path_gc(gc) {}
};

using PdtShortestPathArgs =
    std::tuple<const FstClass &, const std::vector<LabelPair> &,
               MutableFstClass *, const PdtShortestPathOptions &>;

template <class Arc>
void PdtShortestPath(PdtShortestPathArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<2>(*args)->GetMutableFst<Arc>();
  const PdtShortestPathOptions &opts = std::get<3>(*args);
  // In case Arc::Label is not the same as FstClass::Label, we make a
  // copy. Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  switch (opts.queue_type) {
    default:
      FSTERROR() << "Unknown queue type: " << opts.queue_type;
    case FIFO_QUEUE: {
      using Queue = FifoQueue<typename Arc::StateId>;
      fst::PdtShortestPathOptions<Arc, Queue> spopts(opts.keep_parentheses,
                                                         opts.path_gc);
      ShortestPath(fst, typed_parens, ofst, spopts);
      return;
    }
    case LIFO_QUEUE: {
      using Queue = LifoQueue<typename Arc::StateId>;
      fst::PdtShortestPathOptions<Arc, Queue> spopts(opts.keep_parentheses,
                                                         opts.path_gc);
      ShortestPath(fst, typed_parens, ofst, spopts);
      return;
    }
    case STATE_ORDER_QUEUE: {
      using Queue = StateOrderQueue<typename Arc::StateId>;
      fst::PdtShortestPathOptions<Arc, Queue> spopts(opts.keep_parentheses,
                                                         opts.path_gc);
      ShortestPath(fst, typed_parens, ofst, spopts);
      return;
    }
  }
}

void PdtShortestPath(const FstClass &ifst,
    const std::vector<LabelPair> &parens, MutableFstClass *ofst,
    const PdtShortestPathOptions &opts = PdtShortestPathOptions());

// PRINT INFO

using PrintPdtInfoArgs =
    std::pair<const FstClass &, const std::vector<LabelPair> &>;

template <class Arc>
void PrintPdtInfo(PrintPdtInfoArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  // In case Arc::Label is not the same as FstClass::Label, we make a
  // copy. Truncation may occur if FstClass::Label has more precision than
  // Arc::Label.
  std::vector<std::pair<typename Arc::Label, typename Arc::Label>> typed_parens(
      std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_parens.begin());
  PdtInfo<Arc> pdtinfo(fst, typed_parens);
  PrintPdtInfo(pdtinfo);
}

void PrintPdtInfo(const FstClass &ifst, const std::vector<LabelPair> &parens);

}  // namespace script
}  // namespace fst

#define REGISTER_FST_PDT_OPERATIONS(ArcType)                             \
  REGISTER_FST_OPERATION(PdtCompose, ArcType, PdtComposeArgs);           \
  REGISTER_FST_OPERATION(PdtExpand, ArcType, PdtExpandArgs);             \
  REGISTER_FST_OPERATION(PdtReplace, ArcType, PdtReplaceArgs);           \
  REGISTER_FST_OPERATION(PdtReverse, ArcType, PdtReverseArgs);           \
  REGISTER_FST_OPERATION(PdtShortestPath, ArcType, PdtShortestPathArgs); \
  REGISTER_FST_OPERATION(PrintPdtInfo, ArcType, PrintPdtInfoArgs)
#endif  // FST_EXTENSIONS_PDT_PDTSCRIPT_H_
