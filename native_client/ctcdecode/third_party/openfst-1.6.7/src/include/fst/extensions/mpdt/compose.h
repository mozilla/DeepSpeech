// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Compose an MPDT and an FST.

#ifndef FST_EXTENSIONS_MPDT_COMPOSE_H_
#define FST_EXTENSIONS_MPDT_COMPOSE_H_

#include <list>

#include <fst/extensions/mpdt/mpdt.h>
#include <fst/extensions/pdt/compose.h>
#include <fst/compose.h>

namespace fst {

template <class Filter>
class MPdtParenFilter {
 public:
  using FST1 = typename Filter::FST1;
  using FST2 = typename Filter::FST2;
  using Arc = typename Filter::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Matcher1 = typename Filter::Matcher1;
  using Matcher2 = typename Filter::Matcher2;

  using StackId = StateId;
  using ParenStack = internal::MPdtStack<StackId, Label>;
  using FilterState1 = typename Filter::FilterState;
  using FilterState2 = IntegerFilterState<StackId>;
  using FilterState = PairFilterState<FilterState1, FilterState2>;

  MPdtParenFilter(const FST1 &fst1, const FST2 &fst2,
                  Matcher1 *matcher1 = nullptr, Matcher2 *matcher2 = nullptr,
                  const std::vector<std::pair<Label, Label>> *parens = nullptr,
                  const std::vector<Label> *assignments = nullptr,
                  bool expand = false, bool keep_parens = true)
      : filter_(fst1, fst2, matcher1, matcher2),
        parens_(parens ? *parens : std::vector<std::pair<Label, Label>>()),
        assignments_(assignments ? *assignments : std::vector<Label>()),
        expand_(expand),
        keep_parens_(keep_parens),
        fs_(FilterState::NoState()),
        stack_(parens_, assignments_),
        paren_id_(-1) {
    if (parens) {
      for (const auto &pair : *parens) {
        parens_.push_back(pair);
        GetMatcher1()->AddOpenParen(pair.first);
        GetMatcher2()->AddOpenParen(pair.first);
        if (!expand_) {
          GetMatcher1()->AddCloseParen(pair.second);
          GetMatcher2()->AddCloseParen(pair.second);
        }
      }
    }
  }

  MPdtParenFilter(const MPdtParenFilter &filter, bool safe = false)
      : filter_(filter.filter_, safe),
        parens_(filter.parens_),
        expand_(filter.expand_),
        keep_parens_(filter.keep_parens_),
        fs_(FilterState::NoState()),
        stack_(filter.parens_, filter.assignments_),
        paren_id_(-1) {}

  FilterState Start() const {
    return FilterState(filter_.Start(), FilterState2(0));
  }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    fs_ = fs;
    filter_.SetState(s1, s2, fs_.GetState1());
    if (!expand_) return;
    const auto paren_id = stack_.Top(fs.GetState2().GetState());
    if (paren_id != paren_id_) {
      if (paren_id_ != -1) {
        GetMatcher1()->RemoveCloseParen(parens_[paren_id_].second);
        GetMatcher2()->RemoveCloseParen(parens_[paren_id_].second);
      }
      paren_id_ = paren_id;
      if (paren_id_ != -1) {
        GetMatcher1()->AddCloseParen(parens_[paren_id_].second);
        GetMatcher2()->AddCloseParen(parens_[paren_id_].second);
      }
    }
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    const auto fs1 = filter_.FilterArc(arc1, arc2);
    const auto &fs2 = fs_.GetState2();
    if (fs1 == FilterState1::NoState()) return FilterState::NoState();
    if (arc1->olabel == kNoLabel && arc2->ilabel) {  // arc2 parentheses.
      if (keep_parens_) {
        arc1->ilabel = arc2->ilabel;
      } else if (arc2->ilabel) {
        arc2->olabel = arc1->ilabel;
      }
      return FilterParen(arc2->ilabel, fs1, fs2);
    } else if (arc2->ilabel == kNoLabel && arc1->olabel) {  // arc1 parentheses
      if (keep_parens_) {
        arc2->olabel = arc1->olabel;
      } else {
        arc1->ilabel = arc2->olabel;
      }
      return FilterParen(arc1->olabel, fs1, fs2);
    } else {
      return FilterState(fs1, fs2);
    }
  }

  void FilterFinal(Weight *w1, Weight *w2) const {
    if (fs_.GetState2().GetState() != 0) *w1 = Weight::Zero();
    filter_.FilterFinal(w1, w2);
  }

  // Returns respective matchers; ownership stays with filter.

  Matcher1 *GetMatcher1() { return filter_.GetMatcher1(); }

  Matcher2 *GetMatcher2() { return filter_.GetMatcher2(); }

  uint64 Properties(uint64 iprops) const {
    const auto oprops = filter_.Properties(iprops);
    return oprops & kILabelInvariantProperties & kOLabelInvariantProperties;
  }

 private:
  const FilterState FilterParen(Label label, const FilterState1 &fs1,
                                const FilterState2 &fs2) const {
    if (!expand_) return FilterState(fs1, fs2);
    const auto stack_id = stack_.Find(fs2.GetState(), label);
    if (stack_id < 0) {
      return FilterState::NoState();
    } else {
      return FilterState(fs1, FilterState2(stack_id));
    }
  }

  Filter filter_;
  std::vector<std::pair<Label, Label>> parens_;
  std::vector<Label> assignments_;
  bool expand_;       // Expands to FST?
  bool keep_parens_;  // Retains parentheses in output?
  FilterState fs_;    // Current filter state.
  mutable ParenStack stack_;
  ssize_t paren_id_;
};

// Class to setup composition options for MPDT composition. Default is to take
// the MPDT as the first composition argument.
template <class Arc, bool left_pdt = true>
class MPdtComposeFstOptions
    : public ComposeFstOptions<Arc, ParenMatcher<Fst<Arc>>,
                               MPdtParenFilter<AltSequenceComposeFilter<
                                   ParenMatcher<Fst<Arc>> >> > {
 public:
  using Label = typename Arc::Label;
  using MPdtMatcher = ParenMatcher<Fst<Arc>>;
  using MPdtFilter = MPdtParenFilter<AltSequenceComposeFilter<MPdtMatcher>>;

  using ComposeFstOptions<Arc, MPdtMatcher, MPdtFilter>::matcher1;
  using ComposeFstOptions<Arc, MPdtMatcher, MPdtFilter>::matcher2;
  using ComposeFstOptions<Arc, MPdtMatcher, MPdtFilter>::filter;

  MPdtComposeFstOptions(const Fst<Arc> &ifst1,
                        const std::vector<std::pair<Label, Label>> &parens,
                        const std::vector<typename Arc::Label> &assignments,
                        const Fst<Arc> &ifst2, bool expand = false,
                        bool keep_parens = true) {
    matcher1 = new MPdtMatcher(ifst1, MATCH_OUTPUT, kParenList);
    matcher2 = new MPdtMatcher(ifst2, MATCH_INPUT, kParenLoop);
    filter = new MPdtFilter(ifst1, ifst2, matcher1, matcher2, &parens,
                            &assignments, expand, keep_parens);
  }
};

// Class to setup composition options for PDT with FST composition.
// Specialization is for the FST as the first composition argument.
template <class Arc>
class MPdtComposeFstOptions<Arc, false>
    : public ComposeFstOptions<
          Arc, ParenMatcher<Fst<Arc>>,
          MPdtParenFilter<SequenceComposeFilter<ParenMatcher<Fst<Arc>> >> > {
 public:
  using Label = typename Arc::Label;
  using MPdtMatcher = ParenMatcher<Fst<Arc>>;
  using MPdtFilter = MPdtParenFilter<SequenceComposeFilter<MPdtMatcher>>;

  using ComposeFstOptions<Arc, MPdtMatcher, MPdtFilter>::matcher1;
  using ComposeFstOptions<Arc, MPdtMatcher, MPdtFilter>::matcher2;
  using ComposeFstOptions<Arc, MPdtMatcher, MPdtFilter>::filter;

  MPdtComposeFstOptions(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                        const std::vector<std::pair<Label, Label>> &parens,
                        const std::vector<typename Arc::Label> &assignments,
                        bool expand = false, bool keep_parens = true) {
    matcher1 = new MPdtMatcher(ifst1, MATCH_OUTPUT, kParenLoop);
    matcher2 = new MPdtMatcher(ifst2, MATCH_INPUT, kParenList);
    filter = new MPdtFilter(ifst1, ifst2, matcher1, matcher2, &parens,
                            &assignments, expand, keep_parens);
  }
};

struct MPdtComposeOptions {
  bool connect;                  // Connect output?
  PdtComposeFilter filter_type;  // Which pre-defined filter to use.

  explicit MPdtComposeOptions(bool connect = true,
                              PdtComposeFilter filter_type = PAREN_FILTER)
      : connect(connect), filter_type(filter_type) {}
};

// Composes multi-pushdown transducer (MPDT) encoded as an FST (1st arg) and an
// FST (2nd arg) with the result also an MPDT encoded as an FST (3rd arg). In
// theMPDTs, some transitions are labeled with open or close parentheses (and
// associated with a stack). To be interpreted as an MPDT, the parents on each
// stack must balance on a path (see MPdtExpand()). The open-close parenthesis
// label pairs are passed using the parens arguments, and the stack assignments
// are passed using the assignments argument.
template <class Arc>
void Compose(
    const Fst<Arc> &ifst1,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens,
    const std::vector<typename Arc::Label> &assignments, const Fst<Arc> &ifst2,
    MutableFst<Arc> *ofst,
    const MPdtComposeOptions &opts = MPdtComposeOptions()) {
  bool expand = opts.filter_type != PAREN_FILTER;
  bool keep_parens = opts.filter_type != EXPAND_FILTER;
  MPdtComposeFstOptions<Arc, true> copts(ifst1, parens, assignments, ifst2,
                                         expand, keep_parens);
  copts.gc_limit = 0;
  *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
  if (opts.connect) Connect(ofst);
}

// Composes an FST (1st arg) and a multi-pushdown transducer (MPDT) encoded as
// an FST (2nd arg) with the result also an MPDT encoded as an FST (3rd arg).
// In the MPDTs, some transitions are labeled with open or close parentheses
// (and associated with a stack). To be interpreted as an MPDT, the parents on
// each stack must balance on a path (see MPdtExpand()). The open-close
// parenthesis label pairs are passed using the parens arguments, and the stack
// assignments are passed using the assignments argument.
template <class Arc>
void Compose(
    const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens,
    const std::vector<typename Arc::Label> &assignments, MutableFst<Arc> *ofst,
    const MPdtComposeOptions &opts = MPdtComposeOptions()) {
  bool expand = opts.filter_type != PAREN_FILTER;
  bool keep_parens = opts.filter_type != EXPAND_FILTER;
  MPdtComposeFstOptions<Arc, false> copts(ifst1, ifst2, parens, assignments,
                                          expand, keep_parens);
  copts.gc_limit = 0;
  *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
  if (opts.connect) Connect(ofst);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_MPDT_COMPOSE_H_
