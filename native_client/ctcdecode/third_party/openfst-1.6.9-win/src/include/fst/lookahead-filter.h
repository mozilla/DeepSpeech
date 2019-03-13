// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Composition filters to support lookahead matchers, useful for improving
// composition efficiency with certain inputs.

#ifndef FST_LOOKAHEAD_FILTER_H_
#define FST_LOOKAHEAD_FILTER_H_

#include <vector>

#include <fst/log.h>

#include <fst/filter-state.h>
#include <fst/fst.h>
#include <fst/lookahead-matcher.h>


namespace fst {

// Identifies and verifies the capabilities of the matcher to be used for
// lookahead with the composition filters below. This version is passed two
// matchers.
template <class Matcher1, class Matcher2>
MatchType LookAheadMatchType(const Matcher1 &m1, const Matcher2 &m2) {
  const auto type1 = m1.Type(false);
  const auto type2 = m2.Type(false);
  if (type1 == MATCH_OUTPUT && m1.Flags() & kOutputLookAheadMatcher) {
    return MATCH_OUTPUT;
  } else if (type2 == MATCH_INPUT && m2.Flags() & kInputLookAheadMatcher) {
    return MATCH_INPUT;
  } else if (m1.Flags() & kOutputLookAheadMatcher &&
             m1.Type(true) == MATCH_OUTPUT) {
    return MATCH_OUTPUT;
  } else if (m2.Flags() & kInputLookAheadMatcher &&
             m2.Type(true) == MATCH_INPUT) {
    return MATCH_INPUT;
  } else {
    return MATCH_NONE;
  }
}

// Identifies and verifies the capabilities of the matcher to be used for
// lookahead with the composition filters below. This version uses the FST's
// default matchers.
template <class Arc>
MatchType LookAheadMatchType(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {
  LookAheadMatcher<Fst<Arc>> matcher1(fst1, MATCH_OUTPUT);
  LookAheadMatcher<Fst<Arc>> matcher2(fst2, MATCH_INPUT);
  return LookAheadMatchType(matcher1, matcher2);
}

// LookAheadSelector is a helper class for selecting among possibly distinct
// FST and matcher types without using a common base class. This lets us avoid
// virtual function calls. It stores and returns the appropriate FSTs and
// matcher for lookahead. It is templated on the matcher types. General case
// has no methods.
template <class Matcher1, class Matcher2, MatchType MT>
class LookAheadSelector {};

// Stores and returns the appropriate FST and matcher for lookahead. Specialized
// for two matchers of same type with the (match) type argument determining
// which is used for lookahead.
template <class Matcher, MatchType MT>
class LookAheadSelector<Matcher, Matcher, MT> {
 public:
  using FST = typename Matcher::FST;

  LookAheadSelector(Matcher *lmatcher1, Matcher *lmatcher2, MatchType type)
      : lmatcher1_(lmatcher1->Copy()),
        lmatcher2_(lmatcher2->Copy()),
        type_(type) {}

  LookAheadSelector(const LookAheadSelector<Matcher, Matcher, MT> &selector)
      : lmatcher1_(selector.lmatcher1_->Copy()),
        lmatcher2_(selector.lmatcher2_->Copy()),
        type_(selector.type_) {}

  const FST &GetFst() const {
    return type_ == MATCH_OUTPUT ? lmatcher2_->GetFst() : lmatcher1_->GetFst();
  }

  Matcher *GetMatcher() const {
    return type_ == MATCH_OUTPUT ? lmatcher1_.get() : lmatcher2_.get();
  }

 private:
  std::unique_ptr<Matcher> lmatcher1_;
  std::unique_ptr<Matcher> lmatcher2_;
  MatchType type_;
};

// Stores and returns the appropriate FST and matcher for lookahead.
// Specialized for lookahead on input labels.
template <class Matcher1, class Matcher2>
class LookAheadSelector<Matcher1, Matcher2, MATCH_INPUT> {
 public:
  using FST1 = typename Matcher1::FST;

  LookAheadSelector(Matcher1 *lmatcher1, Matcher2 *lmatcher2, MatchType)
      : fst_(lmatcher1->GetFst().Copy()), lmatcher_(lmatcher2->Copy()) {}

  LookAheadSelector(
      const LookAheadSelector<Matcher1, Matcher2, MATCH_INPUT> &selector)
      : fst_(selector.fst_->Copy()), lmatcher_(selector.lmatcher_->Copy()) {}

  const FST1 &GetFst() const { return *fst_; }

  Matcher2 *GetMatcher() const { return lmatcher_.get(); }

 private:
  std::unique_ptr<const FST1> fst_;
  std::unique_ptr<Matcher2> lmatcher_;
};

// Stores and returns the appropriate FST and matcher for lookahead.
// Specialized for lookahead on output labels.
template <class Matcher1, class Matcher2>
class LookAheadSelector<Matcher1, Matcher2, MATCH_OUTPUT> {
 public:
  using FST2 = typename Matcher2::FST;

  LookAheadSelector(Matcher1 *lmatcher1, Matcher2 *lmatcher2, MatchType)
      : fst_(lmatcher2->GetFst().Copy()), lmatcher_(lmatcher1->Copy()) {}

  LookAheadSelector(
      const LookAheadSelector<Matcher1, Matcher2, MATCH_OUTPUT> &selector)
      : fst_(selector.fst_->Copy()), lmatcher_(selector.lmatcher_->Copy()) {}

  const FST2 &GetFst() const { return *fst_; }

  Matcher1 *GetMatcher() const { return lmatcher_.get(); }

 private:
  std::unique_ptr<const FST2> fst_;
  std::unique_ptr<Matcher1> lmatcher_;
};

// This filter uses a lookahead matcher in FilterArc(arc1, arc2) to examine the
// future of the composition state (arc1.nextstate, arc2.nextstate), blocking
// moving forward when its determined to be
// non-coaccessible. It is templated on an underlying filter, typically the
// epsilon filter. Which matcher is the lookahead matcher is determined by the
// template argument MT unless it is MATCH_BOTH. In that case, both matcher
// arguments must be lookahead matchers of the same type and one will be
// selected by LookAheadMatchType() based on their capability.
template <class Filter, class M1 = LookAheadMatcher<typename Filter::FST1>,
          class M2 = M1, MatchType MT = MATCH_BOTH>
class LookAheadComposeFilter {
 public:
  using Arc = typename Filter::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FST1 = typename Filter::FST1;
  using FST2 = typename Filter::FST2;
  using Matcher1 = typename Filter::Matcher1;
  using Matcher2 = typename Filter::Matcher2;
  using FilterState = typename Filter::FilterState;

  LookAheadComposeFilter(const FST1 &fst1, const FST2 &fst2, M1 *matcher1,
                         M2 *matcher2)
      : filter_(fst1, fst2, matcher1, matcher2),
        lookahead_type_(MT == MATCH_BOTH
                            ? LookAheadMatchType(*filter_.GetMatcher1(),
                                                 *filter_.GetMatcher2())
                            : MT),
        selector_(filter_.GetMatcher1(), filter_.GetMatcher2(),
                  lookahead_type_),
        flags_(lookahead_type_ == MATCH_OUTPUT
                   ? filter_.GetMatcher1()->Flags()
                   : filter_.GetMatcher2()->Flags()) {
    if (lookahead_type_ == MATCH_NONE) {
      FSTERROR() << "LookAheadComposeFilter: 1st argument cannot "
                 << "match/look-ahead on output labels and 2nd argument "
                 << "cannot match/look-ahead on input labels";
    }
    selector_.GetMatcher()->InitLookAheadFst(selector_.GetFst());
  }

  LookAheadComposeFilter(
      const LookAheadComposeFilter<Filter, M1, M2, MT> &filter,
      bool safe = false)
      : filter_(filter.filter_, safe),
        lookahead_type_(filter.lookahead_type_),
        selector_(filter_.GetMatcher1(), filter_.GetMatcher2(),
                  lookahead_type_),
        flags_(filter.flags_) {
    selector_.GetMatcher()->InitLookAheadFst(selector_.GetFst(), true);
  }

  FilterState Start() const { return filter_.Start(); }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    filter_.SetState(s1, s2, fs);
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    lookahead_arc_ = false;
    const FilterState &fs = filter_.FilterArc(arc1, arc2);
    if (fs == FilterState::NoState()) return FilterState::NoState();
    return LookAheadOutput() ? LookAheadFilterArc(arc1, arc2, fs)
                             : LookAheadFilterArc(arc2, arc1, fs);
  }

  void FilterFinal(Weight *weight1, Weight *weight2) const {
    filter_.FilterFinal(weight1, weight2);
  }

  // Returns matchers; ownership stays with filter.

  Matcher1 *GetMatcher1() { return filter_.GetMatcher1(); }

  Matcher2 *GetMatcher2() { return filter_.GetMatcher2(); }

  const LookAheadSelector<Matcher1, Matcher2, MT> &Selector() const {
    return selector_;
  }

  uint64_t Properties(uint64_t inprops) const {
    auto outprops = filter_.Properties(inprops);
    if (lookahead_type_ == MATCH_NONE) outprops |= kError;
    return outprops;
  }

  uint32_t LookAheadFlags() const { return flags_; }

  bool LookAheadArc() const { return lookahead_arc_; }

  bool LookAheadOutput() const {
    if (MT == MATCH_OUTPUT) {
      return true;
    } else if (MT == MATCH_INPUT) {
      return false;
    } else if (lookahead_type_ == MATCH_OUTPUT) {
      return true;
    } else {
      return false;
    }
  }

 private:
  FilterState LookAheadFilterArc(Arc *arca, Arc *arcb,
                                 const FilterState &fs) const {
    auto &labela = LookAheadOutput() ? arca->olabel : arca->ilabel;
    if (labela != 0 && !(flags_ & kLookAheadNonEpsilons)) return fs;
    if (labela == 0 && !(flags_ & kLookAheadEpsilons)) return fs;
    lookahead_arc_ = true;
    selector_.GetMatcher()->SetState(arca->nextstate);
    return selector_.GetMatcher()->LookAheadFst(selector_.GetFst(),
                                                arcb->nextstate)
               ? fs
               : FilterState::NoState();
  }

  Filter filter_;             // Underlying filter.
  MatchType lookahead_type_;  // Lookahead match type.
  LookAheadSelector<Matcher1, Matcher2, MT> selector_;
  uint32_t flags_;                // Lookahead flags.
  mutable bool lookahead_arc_;  // Look-ahead performed at last FilterArc()?

  LookAheadComposeFilter &operator=(const LookAheadComposeFilter &) = delete;
};

// This filter adds weight-pushing to a lookahead composition filter using the
// LookAheadWeight() method of matcher argument. It is templated on an
// underlying lookahead filter, typically the basic lookahead filter.
// Weight-pushing in composition brings weights forward as much as possible
// based on the lookahead information.
template <class Filter, class M1 = LookAheadMatcher<typename Filter::FST1>,
          class M2 = M1, MatchType MT = MATCH_BOTH>
class PushWeightsComposeFilter {
 public:
  using Arc = typename Filter::Arc;
  using StateId = typename Filter::StateId;
  using Weight = typename Filter::Weight;

  using FST1 = typename Filter::FST1;
  using FST2 = typename Filter::FST2;
  using Matcher1 = typename Filter::Matcher1;
  using Matcher2 = typename Filter::Matcher2;

  using FilterState1 = typename Filter::FilterState;
  using FilterState2 = WeightFilterState<Weight>;
  using FilterState = PairFilterState<FilterState1, FilterState2>;

  PushWeightsComposeFilter(const FST1 &fst1, const FST2 &fst2, M1 *matcher1,
                           M2 *matcher2)
      : filter_(fst1, fst2, matcher1, matcher2), fs_(FilterState::NoState()) {}

  PushWeightsComposeFilter(
      const PushWeightsComposeFilter<Filter, M1, M2, MT> &filter,
      bool safe = false)
      : filter_(filter.filter_, safe), fs_(FilterState::NoState()) {}

  FilterState Start() const {
    return FilterState(filter_.Start(), FilterState2(Weight::One()));
  }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    fs_ = fs;
    filter_.SetState(s1, s2, fs.GetState1());
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    const auto &fs1 = filter_.FilterArc(arc1, arc2);
    if (fs1 == FilterState1::NoState()) return FilterState::NoState();
    if (!(LookAheadFlags() & kLookAheadWeight)) {
      return FilterState(fs1, FilterState2(Weight::One()));
    }
    const auto &lweight = filter_.LookAheadArc()
                              ? Selector().GetMatcher()->LookAheadWeight()
                              : Weight::One();
    const auto &fs2 = fs_.GetState2();
    const auto &fweight = fs2.GetWeight();
    // Disallows Zero() weight futures.
    if (lweight == Weight::Zero()) return FilterState::NoState();
    arc2->weight = Divide(Times(arc2->weight, lweight), fweight);
    return FilterState(fs1, FilterState2(lweight.Quantize()));
  }

  void FilterFinal(Weight *weight1, Weight *weight2) const {
    filter_.FilterFinal(weight1, weight2);
    if (!(LookAheadFlags() & kLookAheadWeight) || *weight1 == Weight::Zero()) {
      return;
    }
    const auto &fs2 = fs_.GetState2();
    const auto &fweight = fs2.GetWeight();
    *weight1 = Divide(*weight1, fweight);
  }

  // Returns matchers; ownership states with filter.

  Matcher1 *GetMatcher1() { return filter_.GetMatcher1(); }

  Matcher2 *GetMatcher2() { return filter_.GetMatcher2(); }

  const LookAheadSelector<Matcher1, Matcher2, MT> &Selector() const {
    return filter_.Selector();
  }

  uint32_t LookAheadFlags() const { return filter_.LookAheadFlags(); }

  bool LookAheadArc() const { return filter_.LookAheadArc(); }

  bool LookAheadOutput() const { return filter_.LookAheadOutput(); }

  uint64_t Properties(uint64_t props) const {
    return filter_.Properties(props) & kWeightInvariantProperties;
  }

 private:
  Filter filter_;   // Underlying filter.
  FilterState fs_;  // Current filter state.

  PushWeightsComposeFilter &operator=(const PushWeightsComposeFilter &) =
      delete;
};

// This filter adds label-pushing to a lookahead composition filter using the
// LookAheadPrefix() method of the matcher argument. It is templated on an
// underlying filter, typically the basic lookahead or weight-pushing lookahead
// filter. Label-pushing in composition matches labels as early as possible
// based on the lookahead information.
template <class Filter, class M1 = LookAheadMatcher<typename Filter::FST1>,
          class M2 = M1, MatchType MT = MATCH_BOTH>
class PushLabelsComposeFilter {
 public:
  using Arc = typename Filter::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FST1 = typename Filter::FST1;
  using FST2 = typename Filter::FST2;
  using Matcher1 = MultiEpsMatcher<typename Filter::Matcher1>;
  using Matcher2 = MultiEpsMatcher<typename Filter::Matcher2>;
  using FilterState1 = typename Filter::FilterState;
  using FilterState2 = IntegerFilterState<Label>;
  using FilterState = PairFilterState<FilterState1, FilterState2>;

  PushLabelsComposeFilter(const FST1 &fst1, const FST2 &fst2, M1 *matcher1,
                          M2 *matcher2)
      : filter_(fst1, fst2, matcher1, matcher2),
        fs_(FilterState::NoState()),
        fst1_(filter_.GetMatcher1()->GetFst()),
        fst2_(filter_.GetMatcher2()->GetFst()),
        matcher1_(fst1_, MATCH_OUTPUT,
                  filter_.LookAheadOutput() ? kMultiEpsList : kMultiEpsLoop,
                  filter_.GetMatcher1(), false),
        matcher2_(fst2_, MATCH_INPUT,
                  filter_.LookAheadOutput() ? kMultiEpsLoop : kMultiEpsList,
                  filter_.GetMatcher2(), false) {}

  PushLabelsComposeFilter(
      const PushLabelsComposeFilter<Filter, M1, M2, MT> &filter,
      bool safe = false)
      : filter_(filter.filter_, safe),
        fs_(FilterState::NoState()),
        fst1_(filter_.GetMatcher1()->GetFst()),
        fst2_(filter_.GetMatcher2()->GetFst()),
        matcher1_(fst1_, MATCH_OUTPUT,
                  filter_.LookAheadOutput() ? kMultiEpsList : kMultiEpsLoop,
                  filter_.GetMatcher1(), false),
        matcher2_(fst2_, MATCH_INPUT,
                  filter_.LookAheadOutput() ? kMultiEpsLoop : kMultiEpsList,
                  filter_.GetMatcher2(), false) {}

  FilterState Start() const {
    return FilterState(filter_.Start(), FilterState2(kNoLabel));
  }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    fs_ = fs;
    filter_.SetState(s1, s2, fs.GetState1());
    if (!(LookAheadFlags() & kLookAheadPrefix)) return;
    narcsa_ = LookAheadOutput() ? internal::NumArcs(fst1_, s1)
                                : internal::NumArcs(fst2_, s2);
    const auto &fs2 = fs_.GetState2();
    const auto &flabel = fs2.GetState();
    GetMatcher1()->ClearMultiEpsLabels();
    GetMatcher2()->ClearMultiEpsLabels();
    if (flabel != kNoLabel) {                   // Have a lookahead label?
      GetMatcher1()->AddMultiEpsLabel(flabel);  // Yes, make it a multi-epsilon
      GetMatcher2()->AddMultiEpsLabel(flabel);  // label so that it matches the
    }                                           // implicit epsilon arc to be
  }                                             // modified below when pushing.

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    if (!(LookAheadFlags() & kLookAheadPrefix)) {
      return FilterState(filter_.FilterArc(arc1, arc2), FilterState2(kNoLabel));
    }
    const auto &fs2 = fs_.GetState2();
    const auto &flabel = fs2.GetState();
    if (flabel != kNoLabel) {  // Have a lookahead label?
      return LookAheadOutput() ? PushedLabelFilterArc(arc1, arc2, flabel)
                               : PushedLabelFilterArc(arc2, arc1, flabel);
    }
    const auto &fs1 = filter_.FilterArc(arc1, arc2);
    if (fs1 == FilterState1::NoState()) return FilterState::NoState();
    if (!filter_.LookAheadArc())
      return FilterState(fs1, FilterState2(kNoLabel));
    return LookAheadOutput() ? PushLabelFilterArc(arc1, arc2, fs1)
                             : PushLabelFilterArc(arc2, arc1, fs1);
  }

  void FilterFinal(Weight *weight1, Weight *weight2) const {
    filter_.FilterFinal(weight1, weight2);
    if (!(LookAheadFlags() & kLookAheadPrefix) || *weight1 == Weight::Zero()) {
      return;
    }
    const auto &fs2 = fs_.GetState2();
    const auto &flabel = fs2.GetState();
    if (flabel != kNoLabel) *weight1 = Weight::Zero();
  }

  // Returns matchers; ownership states with filter.

  Matcher1 *GetMatcher1() { return &matcher1_; }

  Matcher2 *GetMatcher2() { return &matcher2_; }

  uint64_t Properties(uint64_t iprops) const {
    const auto oprops = filter_.Properties(iprops);
    if (LookAheadOutput()) {
      return oprops & kOLabelInvariantProperties;
    } else {
      return oprops & kILabelInvariantProperties;
    }
  }

 private:
  const LookAheadSelector<typename Filter::Matcher1, typename Filter::Matcher2,
                          MT>
      &Selector() const {
    return filter_.Selector();
  }

  // Consumes an already pushed label.
  FilterState PushedLabelFilterArc(Arc *arca, Arc *arcb, Label flabel) const {
    auto &labela = LookAheadOutput() ? arca->olabel : arca->ilabel;
    const auto &labelb = LookAheadOutput() ? arcb->ilabel : arcb->olabel;
    if (labelb != kNoLabel) {
      return FilterState::NoState();  // Blocks non-(multi-)epsilon label
    } else if (labela == flabel) {
      labela = 0;  // Converts match to multi-epsilon to epsilon.
      return Start();
    } else if (labela == 0) {
      if (narcsa_ == 1) return fs_;  // Takes epsilon, keeping state with label.
      Selector().GetMatcher()->SetState(arca->nextstate);
      if (Selector().GetMatcher()->LookAheadLabel(flabel)) {
        return fs_;  // Takes epsilon, keeping state with label.
      } else {
        return FilterState::NoState();  // Blocks non-coaccessible path.
      }
    } else {
      return FilterState::NoState();  // Blocks mismatch to multi-epsilon label.
    }
  }

  // Pushes a label forward when possible.
  FilterState PushLabelFilterArc(Arc *arca, Arc *arcb,
                                 const FilterState1 &fs1) const {
    auto &labela = LookAheadOutput() ? arca->olabel : arca->ilabel;
    const auto &labelb = LookAheadOutput() ? arcb->olabel : arcb->ilabel;
    if (labelb != 0) {  // No place to push.
      return FilterState(fs1, FilterState2(kNoLabel));
    }
    if (labela != 0 &&  // Wrong lookahead prefix type?
        LookAheadFlags() & kLookAheadNonEpsilonPrefix) {
      return FilterState(fs1, FilterState2(kNoLabel));
    }
    Arc larc(kNoLabel, kNoLabel, Weight::Zero(), kNoStateId);
    if (Selector().GetMatcher()->LookAheadPrefix(&larc)) {  // Have prefix arc?
      labela = LookAheadOutput() ? larc.ilabel : larc.olabel;
      arcb->ilabel = larc.ilabel;  // Goes forward on that arc,
      arcb->olabel = larc.olabel;  // thus pushing the label.
      arcb->weight = Times(arcb->weight, larc.weight);
      arcb->nextstate = larc.nextstate;
      return FilterState(fs1, FilterState2(labela));
    } else {
      return FilterState(fs1, FilterState2(kNoLabel));
    }
  }

  uint32_t LookAheadFlags() const { return filter_.LookAheadFlags(); }

  bool LookAheadArc() const { return filter_.LookAheadArc(); }

  bool LookAheadOutput() const { return filter_.LookAheadOutput(); }

  Filter filter_;   // Underlying filter.
  FilterState fs_;  // Current filter state.
  const FST1 &fst1_;
  const FST2 &fst2_;
  Matcher1 matcher1_;  // Multi-epsilon matcher for fst1_.
  Matcher2 matcher2_;  // Multi-epsilon matcher for fst2_.
  std::ptrdiff_t narcsa_;     // Number of arcs leaving look-ahead match FST.

  PushLabelsComposeFilter &operator=(const PushLabelsComposeFilter &) = delete;
};

// Convenience class for setting up composition with a default lookahead matcher
// and filter.
template <class Arc, MatchType type>
class DefaultLookAhead {
 public:
  using M = Matcher<Fst<Arc>>;
  using ComposeFilter = SequenceComposeFilter<M>;
  using FstMatcher = M;
};

// Specializes for MATCH_INPUT to allow lookahead.
template <class Arc>
class DefaultLookAhead<Arc, MATCH_INPUT> {
 public:
  using M = LookAheadMatcher<Fst<Arc>>;
  using SF = SequenceComposeFilter<M>;
  using ComposeFilter = LookAheadComposeFilter<SF, M>;
  using FstMatcher = M;
};

// Specializes for MATCH_OUTPUT to allow lookahead.
template <class Arc>
class DefaultLookAhead<Arc, MATCH_OUTPUT> {
 public:
  using M = LookAheadMatcher<Fst<Arc>>;
  using SF = AltSequenceComposeFilter<M>;
  using ComposeFilter = LookAheadComposeFilter<SF, M>;
  using FstMatcher = M;
};

// Specializes for StdArc to allow weight and label pushing.
template <>
class DefaultLookAhead<StdArc, MATCH_INPUT> {
 public:
  using M = LookAheadMatcher<Fst<StdArc>>;
  using SF = SequenceComposeFilter<M>;
  using LF = LookAheadComposeFilter<SF, M>;
  using WF = PushWeightsComposeFilter<LF, M>;
  using ComposeFilter = PushLabelsComposeFilter<WF, M>;
  using FstMatcher = M;
};

// Specializes for StdArc to allow weight and label pushing.
template <>
class DefaultLookAhead<StdArc, MATCH_OUTPUT> {
 public:
  using M = LookAheadMatcher<Fst<StdArc>>;
  using SF = AltSequenceComposeFilter<M>;
  using LF = LookAheadComposeFilter<SF, M>;
  using WF = PushWeightsComposeFilter<LF, M>;
  using ComposeFilter = PushLabelsComposeFilter<WF, M>;
  using FstMatcher = M;
};

// Specializes for LogArc to allow weight and label pushing.
template <>
class DefaultLookAhead<LogArc, MATCH_INPUT> {
 public:
  using M = LookAheadMatcher<Fst<LogArc>>;
  using SF = SequenceComposeFilter<M>;
  using LF = LookAheadComposeFilter<SF, M>;
  using WF = PushWeightsComposeFilter<LF, M>;
  using ComposeFilter = PushLabelsComposeFilter<WF, M>;
  using FstMatcher = M;
};

// Specializes for LogArc to allow weight and label pushing.
template <>
class DefaultLookAhead<LogArc, MATCH_OUTPUT> {
 public:
  using M = LookAheadMatcher<Fst<LogArc>>;
  using SF = AltSequenceComposeFilter<M>;
  using LF = LookAheadComposeFilter<SF, M>;
  using WF = PushWeightsComposeFilter<LF, M>;
  using ComposeFilter = PushLabelsComposeFilter<WF, M>;
  using FstMatcher = M;
};

}  // namespace fst

#endif  // FST_LOOKAHEAD_FILTER_H_
