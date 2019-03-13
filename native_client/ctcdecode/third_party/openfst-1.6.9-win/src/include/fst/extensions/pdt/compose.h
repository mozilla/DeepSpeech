// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Composes a PDT and an FST.

#ifndef FST_EXTENSIONS_PDT_COMPOSE_H_
#define FST_EXTENSIONS_PDT_COMPOSE_H_

#include <list>

#include <fst/extensions/pdt/pdt.h>
#include <fst/compose.h>

namespace fst {

// Returns paren arcs for Find(kNoLabel).
constexpr uint32_t kParenList = 0x00000001;

// Returns a kNolabel loop for Find(paren).
constexpr uint32_t kParenLoop = 0x00000002;

// This class is a matcher that treats parens as multi-epsilon labels.
// It is most efficient if the parens are in a range non-overlapping with
// the non-paren labels.
template <class F>
class ParenMatcher {
 public:
  using FST = F;
  using M = SortedMatcher<FST>;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST.
  ParenMatcher(const FST &fst, MatchType match_type,
               uint32_t flags = (kParenLoop | kParenList))
      : matcher_(fst, match_type), match_type_(match_type), flags_(flags) {
    if (match_type == MATCH_INPUT) {
      loop_.ilabel = kNoLabel;
      loop_.olabel = 0;
    } else {
      loop_.ilabel = 0;
      loop_.olabel = kNoLabel;
    }
    loop_.weight = Weight::One();
    loop_.nextstate = kNoStateId;
  }

  // This doesn't copy the FST.
  ParenMatcher(const FST *fst, MatchType match_type,
               uint32_t flags = (kParenLoop | kParenList))
      : matcher_(fst, match_type), match_type_(match_type), flags_(flags) {
    if (match_type == MATCH_INPUT) {
      loop_.ilabel = kNoLabel;
      loop_.olabel = 0;
    } else {
      loop_.ilabel = 0;
      loop_.olabel = kNoLabel;
    }
    loop_.weight = Weight::One();
    loop_.nextstate = kNoStateId;
  }

  // This makes a copy of the FST.
  ParenMatcher(const ParenMatcher<FST> &matcher, bool safe = false)
      : matcher_(matcher.matcher_, safe),
        match_type_(matcher.match_type_),
        flags_(matcher.flags_),
        open_parens_(matcher.open_parens_),
        close_parens_(matcher.close_parens_),
        loop_(matcher.loop_) {
    loop_.nextstate = kNoStateId;
  }

  ParenMatcher<FST> *Copy(bool safe = false) const {
    return new ParenMatcher<FST>(*this, safe);
  }

  MatchType Type(bool test) const { return matcher_.Type(test); }

  void SetState(StateId s) {
    matcher_.SetState(s);
    loop_.nextstate = s;
  }

  bool Find(Label match_label);

  bool Done() const { return done_; }

  const Arc &Value() const { return paren_loop_ ? loop_ : matcher_.Value(); }

  void Next();

  Weight Final(StateId s) { return matcher_.Final(s); }

  std::ptrdiff_t Priority(StateId s) { return matcher_.Priority(s); }

  const FST &GetFst() const { return matcher_.GetFst(); }

  uint64_t Properties(uint64_t props) const { return matcher_.Properties(props); }

  uint32_t Flags() const { return matcher_.Flags(); }

  void AddOpenParen(Label label) {
    if (label == 0) {
      FSTERROR() << "ParenMatcher: Bad open paren label: 0";
    } else {
      open_parens_.Insert(label);
    }
  }

  void AddCloseParen(Label label) {
    if (label == 0) {
      FSTERROR() << "ParenMatcher: Bad close paren label: 0";
    } else {
      close_parens_.Insert(label);
    }
  }

  void RemoveOpenParen(Label label) {
    if (label == 0) {
      FSTERROR() << "ParenMatcher: Bad open paren label: 0";
    } else {
      open_parens_.Erase(label);
    }
  }

  void RemoveCloseParen(Label label) {
    if (label == 0) {
      FSTERROR() << "ParenMatcher: Bad close paren label: 0";
    } else {
      close_parens_.Erase(label);
    }
  }

  void ClearOpenParens() { open_parens_.Clear(); }

  void ClearCloseParens() { close_parens_.Clear(); }

  bool IsOpenParen(Label label) const { return open_parens_.Member(label); }

  bool IsCloseParen(Label label) const { return close_parens_.Member(label); }

 private:
  // Advances matcher to next open paren, returning true if it exists.
  bool NextOpenParen();

  // Advances matcher to next close paren, returning true if it exists.
  bool NextCloseParen();

  M matcher_;
  MatchType match_type_;  // Type of match to perform.
  uint32_t flags_;
  // Open paren label set.
  CompactSet<Label, kNoLabel> open_parens_;
  // Close paren label set.
  CompactSet<Label, kNoLabel> close_parens_;
  bool open_paren_list_;   // Matching open paren list?
  bool close_paren_list_;  // Matching close paren list?
  bool paren_loop_;        // Current arc is the implicit paren loop?
  mutable Arc loop_;       // For non-consuming symbols.
  bool done_;              // Matching done?

  ParenMatcher &operator=(const ParenMatcher &) = delete;
};

template <class FST>
inline bool ParenMatcher<FST>::Find(Label match_label) {
  open_paren_list_ = false;
  close_paren_list_ = false;
  paren_loop_ = false;
  done_ = false;
  // Returns all parenthesis arcs.
  if (match_label == kNoLabel && (flags_ & kParenList)) {
    if (open_parens_.LowerBound() != kNoLabel) {
      matcher_.LowerBound(open_parens_.LowerBound());
      open_paren_list_ = NextOpenParen();
      if (open_paren_list_) return true;
    }
    if (close_parens_.LowerBound() != kNoLabel) {
      matcher_.LowerBound(close_parens_.LowerBound());
      close_paren_list_ = NextCloseParen();
      if (close_paren_list_) return true;
    }
  }
  // Returns the implicit paren loop.
  if (match_label > 0 && (flags_ & kParenLoop) &&
      (IsOpenParen(match_label) || IsCloseParen(match_label))) {
    paren_loop_ = true;
    return true;
  }
  // Returns all other labels.
  if (matcher_.Find(match_label)) return true;
  done_ = true;
  return false;
}

template <class FST>
inline void ParenMatcher<FST>::Next() {
  if (paren_loop_) {
    paren_loop_ = false;
    done_ = true;
  } else if (open_paren_list_) {
    matcher_.Next();
    open_paren_list_ = NextOpenParen();
    if (open_paren_list_) return;
    if (close_parens_.LowerBound() != kNoLabel) {
      matcher_.LowerBound(close_parens_.LowerBound());
      close_paren_list_ = NextCloseParen();
      if (close_paren_list_) return;
    }
    done_ = !matcher_.Find(kNoLabel);
  } else if (close_paren_list_) {
    matcher_.Next();
    close_paren_list_ = NextCloseParen();
    if (close_paren_list_) return;
    done_ = !matcher_.Find(kNoLabel);
  } else {
    matcher_.Next();
    done_ = matcher_.Done();
  }
}

// Advances matcher to next open paren, returning true if it exists.
template <class FST>
inline bool ParenMatcher<FST>::NextOpenParen() {
  for (; !matcher_.Done(); matcher_.Next()) {
    Label label = match_type_ == MATCH_INPUT ? matcher_.Value().ilabel
                                             : matcher_.Value().olabel;
    if (label > open_parens_.UpperBound()) return false;
    if (IsOpenParen(label)) return true;
  }
  return false;
}

// Advances matcher to next close paren, returning true if it exists.
template <class FST>
inline bool ParenMatcher<FST>::NextCloseParen() {
  for (; !matcher_.Done(); matcher_.Next()) {
    Label label = match_type_ == MATCH_INPUT ? matcher_.Value().ilabel
                                             : matcher_.Value().olabel;
    if (label > close_parens_.UpperBound()) return false;
    if (IsCloseParen(label)) return true;
  }
  return false;
}

template <class Filter>
class ParenFilter {
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
  using ParenStack = PdtStack<StackId, Label>;
  using FilterState1 = typename Filter::FilterState;
  using FilterState2 = IntegerFilterState<StackId>;
  using FilterState = PairFilterState<FilterState1, FilterState2>;

  ParenFilter(const FST1 &fst1, const FST2 &fst2, Matcher1 *matcher1 = nullptr,
              Matcher2 *matcher2 = nullptr,
              const std::vector<std::pair<Label, Label>> *parens = nullptr,
              bool expand = false, bool keep_parens = true)
      : filter_(fst1, fst2, matcher1, matcher2),
        parens_(parens ? *parens : std::vector<std::pair<Label, Label>>()),
        expand_(expand),
        keep_parens_(keep_parens),
        fs_(FilterState::NoState()),
        stack_(parens_),
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

  ParenFilter(const ParenFilter &filter, bool safe = false)
      : filter_(filter.filter_, safe),
        parens_(filter.parens_),
        expand_(filter.expand_),
        keep_parens_(filter.keep_parens_),
        fs_(FilterState::NoState()),
        stack_(filter.parens_),
        paren_id_(-1) {}

  FilterState Start() const {
    return FilterState(filter_.Start(), FilterState2(0));
  }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    fs_ = fs;
    filter_.SetState(s1, s2, fs_.GetState1());
    if (!expand_) return;
    std::ptrdiff_t paren_id = stack_.Top(fs.GetState2().GetState());
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
    } else if (arc2->ilabel == kNoLabel && arc1->olabel) {  // arc1 parentheses.
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

  uint64_t Properties(uint64_t iprops) const {
    return filter_.Properties(iprops) & kILabelInvariantProperties &
           kOLabelInvariantProperties;
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
  bool expand_;       // Expands to FST?
  bool keep_parens_;  // Retains parentheses in output?
  FilterState fs_;    // Current filter state.
  mutable ParenStack stack_;
  std::ptrdiff_t paren_id_;
};

// Class to setup composition options for PDT composition. Default is to take
// the PDT as the first composition argument.
template <class Arc, bool left_pdt = true>
class PdtComposeFstOptions
    : public ComposeFstOptions<
          Arc, ParenMatcher<Fst<Arc>>,
          ParenFilter<AltSequenceComposeFilter<ParenMatcher<Fst<Arc>>>>> {
 public:
  using Label = typename Arc::Label;
  using PdtMatcher = ParenMatcher<Fst<Arc>>;
  using PdtFilter = ParenFilter<AltSequenceComposeFilter<PdtMatcher>>;

  using ComposeFstOptions<Arc, PdtMatcher, PdtFilter>::matcher1;
  using ComposeFstOptions<Arc, PdtMatcher, PdtFilter>::matcher2;
  using ComposeFstOptions<Arc, PdtMatcher, PdtFilter>::filter;

  PdtComposeFstOptions(const Fst<Arc> &ifst1,
                       const std::vector<std::pair<Label, Label>> &parens,
                       const Fst<Arc> &ifst2, bool expand = false,
                       bool keep_parens = true) {
    matcher1 = new PdtMatcher(ifst1, MATCH_OUTPUT, kParenList);
    matcher2 = new PdtMatcher(ifst2, MATCH_INPUT, kParenLoop);
    filter = new PdtFilter(ifst1, ifst2, matcher1, matcher2, &parens, expand,
                           keep_parens);
  }
};

// Class to setup composition options for PDT with FST composition.
// Specialization is for the FST as the first composition argument.
template <class Arc>
class PdtComposeFstOptions<Arc, false>
    : public ComposeFstOptions<
          Arc, ParenMatcher<Fst<Arc>>,
          ParenFilter<SequenceComposeFilter<ParenMatcher<Fst<Arc>>>>> {
 public:
  using Label = typename Arc::Label;
  using PdtMatcher = ParenMatcher<Fst<Arc>>;
  using PdtFilter = ParenFilter<SequenceComposeFilter<PdtMatcher>>;

  using ComposeFstOptions<Arc, PdtMatcher, PdtFilter>::matcher1;
  using ComposeFstOptions<Arc, PdtMatcher, PdtFilter>::matcher2;
  using ComposeFstOptions<Arc, PdtMatcher, PdtFilter>::filter;

  PdtComposeFstOptions(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                       const std::vector<std::pair<Label, Label>> &parens,
                       bool expand = false, bool keep_parens = true) {
    matcher1 = new PdtMatcher(ifst1, MATCH_OUTPUT, kParenLoop);
    matcher2 = new PdtMatcher(ifst2, MATCH_INPUT, kParenList);
    filter = new PdtFilter(ifst1, ifst2, matcher1, matcher2, &parens, expand,
                           keep_parens);
  }
};

enum PdtComposeFilter {
  PAREN_FILTER,         // Bar-Hillel construction; keeps parentheses.
  EXPAND_FILTER,        // Bar-Hillel + expansion; removes parentheses.
  EXPAND_PAREN_FILTER,  // Bar-Hillel + expansion; keeps parentheses.
};

struct PdtComposeOptions {
  bool connect;                  // Connect output?
  PdtComposeFilter filter_type;  // Pre-defined filter to use.

  explicit PdtComposeOptions(bool connect = true,
                             PdtComposeFilter filter_type = PAREN_FILTER)
      : connect(connect), filter_type(filter_type) {}
};

// Composes pushdown transducer (PDT) encoded as an FST (1st arg) and an FST
// (2nd arg) with the result also a PDT encoded as an FST (3rd arg). In the
// PDTs, some transitions are labeled with open or close parentheses. To be
// interpreted as a PDT, the parens must balance on a path (see PdtExpand()).
// The open-close parenthesis label pairs are passed using the parens argument.
template <class Arc>
void Compose(const Fst<Arc> &ifst1,
             const std::vector<
                 std::pair<typename Arc::Label, typename Arc::Label>> &parens,
             const Fst<Arc> &ifst2, MutableFst<Arc> *ofst,
             const PdtComposeOptions &opts = PdtComposeOptions()) {
  bool expand = opts.filter_type != PAREN_FILTER;
  bool keep_parens = opts.filter_type != EXPAND_FILTER;
  PdtComposeFstOptions<Arc, true> copts(ifst1, parens, ifst2, expand,
                                        keep_parens);
  copts.gc_limit = 0;
  *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
  if (opts.connect) Connect(ofst);
}

// Composes an FST (1st arg) and pushdown transducer (PDT) encoded as an FST
// (2nd arg) with the result also a PDT encoded as an FST (3rd arg). In the
// PDTs, some transitions are labeled with open or close parentheses. To be
// interpreted as a PDT, the parens must balance on a path (see ExpandFst()).
// The open-close parenthesis label pairs are passed using the parens argument.
template <class Arc>
void Compose(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
             const std::vector<
                 std::pair<typename Arc::Label, typename Arc::Label>> &parens,
             MutableFst<Arc> *ofst,
             const PdtComposeOptions &opts = PdtComposeOptions()) {
  bool expand = opts.filter_type != PAREN_FILTER;
  bool keep_parens = opts.filter_type != EXPAND_FILTER;
  PdtComposeFstOptions<Arc, false> copts(ifst1, ifst2, parens, expand,
                                         keep_parens);
  copts.gc_limit = 0;
  *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
  if (opts.connect) Connect(ofst);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_COMPOSE_H_
