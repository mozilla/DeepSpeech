// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes for filtering the composition matches, e.g. for correct epsilon
// handling.

#ifndef FST_COMPOSE_FILTER_H_
#define FST_COMPOSE_FILTER_H_

#include <fst/filter-state.h>
#include <fst/fst-decl.h>  // For optional argument declarations
#include <fst/fst.h>
#include <fst/matcher.h>


namespace fst {

// Composition filters determine which matches are allowed to proceed. The
// filter's state is represeted by the type ComposeFilter::FilterState.
// The basic filters handle correct epsilon matching. Their interface is:
//
// template <class M1, class M2>
// class ComposeFilter {
//  public:
//   using Matcher1 = ...;
//   using Matcher2 = ...;
//   using FST1 = typename M1::FST;
//   using FST2 = typename M2::FST;
//   using FilterState = ...;
//
//   using Arc = typename FST1::Arc;
//   using StateId = typename Arc::StateId;
//   using Weight = typename Arc::Weight;
//
//   // Required constructor.
//   ComposeFilter(const FST1 &fst1, const FST2 &fst2,
//                 M1 *matcher1 = nullptr, M2 *matcher2 = nullptr);
//
//   // If safe=true, the copy is thread-safe. See Fst<>::Copy()
//   // for further doc.
//   ComposeFilter(const ComposeFilter<M1, M2> &filter,
//                 bool safe = false);
//
//   // Return start state of filter.
//   FilterState Start() const;
//
//   // Specifies current composition state.
//   void SetState(StateId s1, StateId s2, const FilterState &fs);
//
//   // Apply filter at current composition state to these transitions. If an
//   // arc label to be matched is kNolabel, then that side does not consume a
//   // symbol. Returns the new filter state or, if disallowed,
//   // FilterState::NoState(). The filter is permitted to modify its inputs
//   // (e.g. for optimization reasons).
//   FilterState FilterArc(Arc *arc1, Arc *arc2) const;

//   // Apply filter at current composition state to these final weights
//   // (cf. superfinal transitions). The filter may modify its inputs
//   // (e.g. for optimization reasons).
//   void FilterFinal(Weight *w1, Weight *w2) const;
//
//   // Return the respective matchers. Ownership stays with filter. These
//   // methods allow the filter to access and possibly modify the compositio
//   // matchers (useful, e.g., with lookahead).
//
//   Matcher1 *GetMatcher1();
//
//   Matcher2 *GetMatcher2();
//
//   // This specifies how the filter affects the composition result properties.
//   It takes as argument the properties that would apply with a trivial
//   // composition filter.
//   uint64 Properties(uint64 props) const;
// };
//
// This filter allows only exact matching of symbols from FST1 with on FST2;
// e.g., no special interpretation of epsilons.
template <class M1, class M2 /* = M1 */>
class NullComposeFilter {
 public:
  using Matcher1 = M1;
  using Matcher2 = M2;
  using FST1 = typename M1::FST;
  using FST2 = typename M2::FST;
  using FilterState = TrivialFilterState;

  using Arc = typename FST1::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  NullComposeFilter(const FST1 &fst1, const FST2 &fst2,
                    Matcher1 *matcher1 = nullptr, Matcher2 *matcher2 = nullptr)
      : matcher1_(matcher1 ? matcher1 : new Matcher1(fst1, MATCH_OUTPUT)),
        matcher2_(matcher2 ? matcher2 : new Matcher2(fst2, MATCH_INPUT)),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()) {}

  NullComposeFilter(const NullComposeFilter<M1, M2> &filter, bool safe = false)
      : matcher1_(filter.matcher1_->Copy(safe)),
        matcher2_(filter.matcher2_->Copy(safe)),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()) {}

  FilterState Start() const { return FilterState(true); }

  void SetState(StateId, StateId, const FilterState &) {}

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    return (arc1->olabel == kNoLabel || arc2->ilabel == kNoLabel)
               ? FilterState::NoState()
               : FilterState(true);
  }

  void FilterFinal(Weight *, Weight *) const {}

  Matcher1 *GetMatcher1() { return matcher1_.get(); }

  Matcher2 *GetMatcher2() { return matcher2_.get(); }

  uint64 Properties(uint64 props) const { return props; }

 private:
  std::unique_ptr<Matcher1> matcher1_;
  std::unique_ptr<Matcher2> matcher2_;
  const FST1 &fst1_;
  const FST2 &fst2_;
};

// This filter allows all epsilon matches, potentially resulting in redundant
// epsilon paths. The use of this filter gives correct results iff one of the
// following conditions hold:
//
//  (1) The semiring is idempotent,
//  (2) the first FST is output-epsilon free, or
//  (3) the second FST is input-epsilon free.
//
// For (1), redundant epsilon paths may be created but won't hurt correctness.
// For (2) and (3), no redundant paths are created.
template <class M1, class M2 /* = M1 */>
class TrivialComposeFilter {
 public:
  using Matcher1 = M1;
  using Matcher2 = M2;
  using FST1 = typename M1::FST;
  using FST2 = typename M2::FST;
  using FilterState = TrivialFilterState;

  using Arc = typename FST1::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  TrivialComposeFilter(const FST1 &fst1, const FST2 &fst2,
                       Matcher1 *matcher1 = nullptr,
                       Matcher2 *matcher2 = nullptr)
      : matcher1_(matcher1 ? matcher1 : new Matcher1(fst1, MATCH_OUTPUT)),
        matcher2_(matcher2 ? matcher2 : new Matcher2(fst2, MATCH_INPUT)),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()) {}

  TrivialComposeFilter(const TrivialComposeFilter<Matcher1, Matcher2> &filter,
                       bool safe = false)
      : matcher1_(filter.matcher1_->Copy(safe)),
        matcher2_(filter.matcher2_->Copy(safe)),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()) {}

  FilterState Start() const { return FilterState(true); }

  void SetState(StateId, StateId, const FilterState &) {}

  FilterState FilterArc(Arc *, Arc *) const { return FilterState(true); }

  void FilterFinal(Weight *, Weight *) const {}

  Matcher1 *GetMatcher1() { return matcher1_.get(); }

  Matcher2 *GetMatcher2() { return matcher2_.get(); }

  uint64 Properties(uint64 props) const { return props; }

 private:
  std::unique_ptr<Matcher1> matcher1_;
  std::unique_ptr<Matcher2> matcher2_;
  const FST1 &fst1_;
  const FST2 &fst2_;
};

// This filter requires epsilons on FST1 to be read before epsilons on FST2.
template <class M1, class M2 /* = M1 */>
class SequenceComposeFilter {
 public:
  using Matcher1 = M1;
  using Matcher2 = M2;
  using FST1 = typename M1::FST;
  using FST2 = typename M2::FST;
  using FilterState = CharFilterState;

  using Arc = typename FST1::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  SequenceComposeFilter(const FST1 &fst1, const FST2 &fst2,
                        Matcher1 *matcher1 = nullptr,
                        Matcher2 *matcher2 = nullptr)
      : matcher1_(matcher1 ? matcher1 : new Matcher1(fst1, MATCH_OUTPUT)),
        matcher2_(matcher2 ? matcher2 : new Matcher2(fst2, MATCH_INPUT)),
        fst1_(matcher1_->GetFst()),
        s1_(kNoStateId),
        s2_(kNoStateId),
        fs_(kNoStateId) {}

  SequenceComposeFilter(const SequenceComposeFilter<Matcher1, Matcher2> &filter,
                        bool safe = false)
      : matcher1_(filter.matcher1_->Copy(safe)),
        matcher2_(filter.matcher2_->Copy(safe)),
        fst1_(matcher1_->GetFst()),
        s1_(kNoStateId),
        s2_(kNoStateId),
        fs_(kNoStateId) {}

  FilterState Start() const { return FilterState(0); }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    if (s1_ == s1 && s2_ == s2 && fs == fs_) return;
    s1_ = s1;
    s2_ = s2;
    fs_ = fs;
    const auto na1 = internal::NumArcs(fst1_, s1);
    const auto ne1 = internal::NumOutputEpsilons(fst1_, s1);
    const bool fin1 = internal::Final(fst1_, s1) != Weight::Zero();
    alleps1_ = na1 == ne1 && !fin1;
    noeps1_ = ne1 == 0;
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    if (arc1->olabel == kNoLabel) {
      return alleps1_ ? FilterState::NoState() : noeps1_ ? FilterState(0)
                                                         : FilterState(1);
    } else if (arc2->ilabel == kNoLabel) {
      return fs_ != FilterState(0) ? FilterState::NoState() : FilterState(0);
    } else {
      return arc1->olabel == 0 ? FilterState::NoState() : FilterState(0);
    }
  }

  void FilterFinal(Weight *, Weight *) const {}

  Matcher1 *GetMatcher1() { return matcher1_.get(); }

  Matcher2 *GetMatcher2() { return matcher2_.get(); }

  uint64 Properties(uint64 props) const { return props; }

 private:
  std::unique_ptr<Matcher1> matcher1_;
  std::unique_ptr<Matcher2> matcher2_;
  const FST1 &fst1_;
  StateId s1_;      // Current fst1_ state.
  StateId s2_;      // Current fst2_ state.
  FilterState fs_;  // Current filter state.
  bool alleps1_;   // Only epsilons (and non-final) leaving s1_?
  bool noeps1_;    // No epsilons leaving s1_?
};

// This filter requires epsilons on FST2 to be read before epsilons on FST1.
template <class M1, class M2 /* = M1 */>
class AltSequenceComposeFilter {
 public:
  using Matcher1 = M1;
  using Matcher2 = M2;
  using FST1 = typename M1::FST;
  using FST2 = typename M2::FST;
  using FilterState = CharFilterState;

  using Arc = typename FST1::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  AltSequenceComposeFilter(const FST1 &fst1, const FST2 &fst2,
                           Matcher1 *matcher1 = nullptr,
                           Matcher2 *matcher2 = nullptr)
      : matcher1_(matcher1 ? matcher1 : new Matcher1(fst1, MATCH_OUTPUT)),
        matcher2_(matcher2 ? matcher2 : new Matcher2(fst2, MATCH_INPUT)),
        fst2_(matcher2_->GetFst()),
        s1_(kNoStateId),
        s2_(kNoStateId),
        fs_(kNoStateId) {}

  AltSequenceComposeFilter(
      const AltSequenceComposeFilter<Matcher1, Matcher2> &filter,
      bool safe = false)
      : matcher1_(filter.matcher1_->Copy(safe)),
        matcher2_(filter.matcher2_->Copy(safe)),
        fst2_(matcher2_->GetFst()),
        s1_(kNoStateId),
        s2_(kNoStateId),
        fs_(kNoStateId) {}

  FilterState Start() const { return FilterState(0); }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    if (s1_ == s1 && s2_ == s2 && fs == fs_) return;
    s1_ = s1;
    s2_ = s2;
    fs_ = fs;
    const auto na2 = internal::NumArcs(fst2_, s2);
    const auto ne2 = internal::NumInputEpsilons(fst2_, s2);
    const bool fin2 = internal::Final(fst2_, s2) != Weight::Zero();
    alleps2_ = na2 == ne2 && !fin2;
    noeps2_ = ne2 == 0;
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    if (arc2->ilabel == kNoLabel) {
      return alleps2_ ? FilterState::NoState() : noeps2_ ? FilterState(0)
                                                         : FilterState(1);
    } else if (arc1->olabel == kNoLabel) {
      return fs_ == FilterState(1) ? FilterState::NoState() : FilterState(0);
    } else {
      return arc1->olabel == 0 ? FilterState::NoState() : FilterState(0);
    }
  }

  void FilterFinal(Weight *, Weight *) const {}

  Matcher1 *GetMatcher1() { return matcher1_.get(); }

  Matcher2 *GetMatcher2() { return matcher2_.get(); }

  uint64 Properties(uint64 props) const { return props; }

 private:
  std::unique_ptr<Matcher1> matcher1_;
  std::unique_ptr<Matcher2> matcher2_;
  const FST2 &fst2_;
  StateId s1_;      // Current fst1_ state.
  StateId s2_;      // Current fst2_ state.
  FilterState fs_;  // Current filter state.
  bool alleps2_;    // Only epsilons (and non-final) leaving s2_?
  bool noeps2_;     // No epsilons leaving s2_?
};

// This filter requires epsilons on FST1 to be matched with epsilons on FST2
// whenever possible. (Template arg default declared in fst-decl.h.)
template <class M1, class M2 /* = M1 */>
class MatchComposeFilter {
 public:
  using Matcher1 = M1;
  using Matcher2 = M2;
  using FST1 = typename M1::FST;
  using FST2 = typename M2::FST;
  using FilterState = CharFilterState;

  using Arc = typename FST1::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  MatchComposeFilter(const FST1 &fst1, const FST2 &fst2,
                     Matcher1 *matcher1 = nullptr, Matcher2 *matcher2 = nullptr)
      : matcher1_(matcher1 ? matcher1 : new Matcher1(fst1, MATCH_OUTPUT)),
        matcher2_(matcher2 ? matcher2 : new Matcher2(fst2, MATCH_INPUT)),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()),
        s1_(kNoStateId),
        s2_(kNoStateId),
        fs_(kNoStateId) {}

  MatchComposeFilter(const MatchComposeFilter<Matcher1, Matcher2> &filter,
                     bool safe = false)
      : matcher1_(filter.matcher1_->Copy(safe)),
        matcher2_(filter.matcher2_->Copy(safe)),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()),
        s1_(kNoStateId),
        s2_(kNoStateId),
        fs_(kNoStateId) {}

  FilterState Start() const { return FilterState(0); }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    if (s1_ == s1 && s2_ == s2 && fs == fs_) return;
    s1_ = s1;
    s2_ = s2;
    fs_ = fs;
    size_t na1 = internal::NumArcs(fst1_, s1);
    size_t ne1 = internal::NumOutputEpsilons(fst1_, s1);
    bool f1 = internal::Final(fst1_, s1) != Weight::Zero();
    alleps1_ = na1 == ne1 && !f1;
    noeps1_ = ne1 == 0;
    size_t na2 = internal::NumArcs(fst2_, s2);
    size_t ne2 = internal::NumInputEpsilons(fst2_, s2);
    bool f2 = internal::Final(fst2_, s2) != Weight::Zero();
    alleps2_ = na2 == ne2 && !f2;
    noeps2_ = ne2 == 0;
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    if (arc2->ilabel == kNoLabel) {  // Epsilon in FST1.
      return fs_ == FilterState(0)
                 ? (noeps2_
                        ? FilterState(0)
                        : (alleps2_ ? FilterState::NoState() : FilterState(1)))
                 : (fs_ == FilterState(1) ? FilterState(1)
                                          : FilterState::NoState());
    } else if (arc1->olabel == kNoLabel) {  // Epsilon in FST2.
      return fs_ == FilterState(0)
                 ? (noeps1_
                        ? FilterState(0)
                        : (alleps1_ ? FilterState::NoState() : FilterState(2)))
                 : (fs_ == FilterState(2) ? FilterState(2)
                                          : FilterState::NoState());
    } else if (arc1->olabel == 0) {  // Epsilon in both.
      return fs_ == FilterState(0) ? FilterState(0) : FilterState::NoState();
    } else {  // Both are non-epsilons.
      return FilterState(0);
    }
  }

  void FilterFinal(Weight *, Weight *) const {}

  Matcher1 *GetMatcher1() { return matcher1_.get(); }

  Matcher2 *GetMatcher2() { return matcher2_.get(); }

  uint64 Properties(uint64 props) const { return props; }

 private:
  std::unique_ptr<Matcher1> matcher1_;
  std::unique_ptr<Matcher2> matcher2_;
  const FST1 &fst1_;
  const FST2 &fst2_;
  StateId s1_;      // Current fst1_ state.
  StateId s2_;      // Current fst2_ state.
  FilterState fs_;  // Current filter state ID.
  bool alleps1_;    // Only epsilson (and non-final) leaving s1?
  bool alleps2_;    // Only epsilons (and non-final) leaving s2?
  bool noeps1_;     // No epsilons leaving s1?
  bool noeps2_;     // No epsilons leaving s2?
};

// This filter works with the MultiEpsMatcher to determine if multi-epsilons are
// preserved in the composition output (rather than rewritten as 0) and
// ensures correct properties.
template <class Filter>
class MultiEpsFilter {
 public:
  using Matcher1 = typename Filter::Matcher1;
  using Matcher2 = typename Filter::Matcher2;
  using FST1 = typename Filter::FST1;
  using FST2 = typename Filter::FST2;
  using FilterState = typename Filter::FilterState;

  using Arc = typename Filter::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  MultiEpsFilter(const FST1 &fst1, const FST2 &fst2,
                 Matcher1 *matcher1 = nullptr, Matcher2 *matcher2 = nullptr,
                 bool keep_multi_eps = false)
      : filter_(fst1, fst2, matcher1, matcher2),
        keep_multi_eps_(keep_multi_eps) {}

  MultiEpsFilter(const MultiEpsFilter<Filter> &filter, bool safe = false)
      : filter_(filter.filter_, safe),
        keep_multi_eps_(filter.keep_multi_eps_) {}

  FilterState Start() const { return filter_.Start(); }

  void SetState(StateId s1, StateId s2, const FilterState &fs) {
    return filter_.SetState(s1, s2, fs);
  }

  FilterState FilterArc(Arc *arc1, Arc *arc2) const {
    const auto fs = filter_.FilterArc(arc1, arc2);
    if (keep_multi_eps_) {
      if (arc1->olabel == kNoLabel) arc1->ilabel = arc2->ilabel;
      if (arc2->ilabel == kNoLabel) arc2->olabel = arc1->olabel;
    }
    return fs;
  }

  void FilterFinal(Weight *w1, Weight *w2) const {
    return filter_.FilterFinal(w1, w2);
  }

  Matcher1 *GetMatcher1() { return filter_.GetMatcher1(); }

  Matcher2 *GetMatcher2() { return filter_.GetMatcher2(); }

  uint64 Properties(uint64 iprops) const {
    const auto oprops = filter_.Properties(iprops);
    return oprops & kILabelInvariantProperties & kOLabelInvariantProperties;
  }

 private:
  Filter filter_;
  bool keep_multi_eps_;
};

}  // namespace fst

#endif  // FST_COMPOSE_FILTER_H_
