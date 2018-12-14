// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to compute the composition of two FSTs.

#ifndef FST_COMPOSE_H_
#define FST_COMPOSE_H_

#include <algorithm>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/compose-filter.h>
#include <fst/fst-decl.h>  // For optional argument declarations
#include <fst/lookahead-filter.h>
#include <fst/matcher.h>
#include <fst/state-table.h>
#include <fst/test-properties.h>


namespace fst {

// Delayed composition options templated on the arc type, the matcher,
// the composition filter, and the composition state table.  By
// default, the matchers, filter, and state table are constructed by
// composition. If set below, the user can instead pass in these
// objects; in that case, ComposeFst takes their ownership. This
// version controls composition implemented between generic Fst<Arc>
// types and a shared matcher type M for Fst<Arc>. This should be
// adequate for most applications, giving a reasonable tradeoff
// between efficiency and code sharing (but see ComposeFstImplOptions).
template <class Arc, class M = Matcher<Fst<Arc>>,
          class Filter = SequenceComposeFilter<M>,
          class StateTable =
              GenericComposeStateTable<Arc, typename Filter::FilterState>>
struct ComposeFstOptions : public CacheOptions {
  M *matcher1;              // FST1 matcher.
  M *matcher2;              // FST2 matcher.
  Filter *filter;           // Composition filter.
  StateTable *state_table;  // Composition state table.

  explicit ComposeFstOptions(const CacheOptions &opts = CacheOptions(),
                             M *matcher1 = nullptr, M *matcher2 = nullptr,
                             Filter *filter = nullptr,
                             StateTable *state_table = nullptr)
      : CacheOptions(opts),
        matcher1(matcher1),
        matcher2(matcher2),
        filter(filter),
        state_table(state_table) {}
};

// Forward declaration of ComposeFstMatcher.
template <class C, class F, class T>
class ComposeFstMatcher;

// Delayed composition options templated on the two matcher types, the
// composition filter, the composition state table and the cache store. By
// default, the matchers, filter, state table and cache store are constructed
// by composition. If set below, the user can instead pass in these objects; in
// that case, ComposeFst takes their ownership. This version controls
// composition implemented using arbitrary matchers (of the same arc type but
// otherwise arbitrary FST type). The user must ensure the matchers are
// compatible. These options permit the most efficient use, but shares the
// least code. This is for advanced use only in the most demanding or
// specialized applications that can benefit from it; otherwise, prefer
// ComposeFstOptions).
template <class M1, class M2, class Filter = SequenceComposeFilter<M1, M2>,
          class StateTable = GenericComposeStateTable<
              typename M1::Arc, typename Filter::FilterState>,
          class CacheStore = DefaultCacheStore<typename M1::Arc>>
struct ComposeFstImplOptions : public CacheImplOptions<CacheStore> {
  M1 *matcher1;    // FST1 matcher (see matcher.h)....
  M2 *matcher2;    // FST2 matcher.
  Filter *filter;  // Composition filter (see compose-filter.h).
  StateTable
    *state_table;        // Composition state table (see compose-state-table.h).
  bool own_state_table;   // ComposeFstImpl takes ownership of 'state_table'?
  bool allow_noncommute;  // Allow non-commutative weights

  explicit ComposeFstImplOptions(const CacheOptions &opts,
                                 M1 *matcher1 = nullptr, M2 *matcher2 = nullptr,
                                 Filter *filter = nullptr,
                                 StateTable *state_table = nullptr)
      : CacheImplOptions<CacheStore>(opts),
        matcher1(matcher1),
        matcher2(matcher2),
        filter(filter),
        state_table(state_table),
        own_state_table(true),
        allow_noncommute(false) {}

  explicit ComposeFstImplOptions(const CacheImplOptions<CacheStore> &opts,
                                 M1 *matcher1 = nullptr, M2 *matcher2 = nullptr,
                                 Filter *filter = nullptr,
                                 StateTable *state_table = nullptr)
      : CacheImplOptions<CacheStore>(opts),
        matcher1(matcher1),
        matcher2(matcher2),
        filter(filter),
        state_table(state_table),
        own_state_table(true),
        allow_noncommute(false) {}

  ComposeFstImplOptions()
      : matcher1(nullptr),
        matcher2(nullptr),
        filter(nullptr),
        state_table(nullptr),
        own_state_table(true),
        allow_noncommute(false) {}
};

namespace internal {

// Implementation of delayed composition. This base class is common to the
// variants with different matchers, composition filters and state tables.
template <class Arc, class CacheStore = DefaultCacheStore<Arc>,
          class F = ComposeFst<Arc, CacheStore>>
class ComposeFstImplBase
    : public CacheBaseImpl<typename CacheStore::State, CacheStore> {
 public:
  using FST = F;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using State = typename CacheStore::State;
  using CacheImpl = CacheBaseImpl<State, CacheStore>;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  using CacheImpl::HasStart;
  using CacheImpl::HasFinal;
  using CacheImpl::HasArcs;
  using CacheImpl::SetFinal;
  using CacheImpl::SetStart;

  ComposeFstImplBase(const CacheImplOptions<CacheStore> &opts)
      : CacheImpl(opts) {}

  ComposeFstImplBase(const CacheOptions &opts) : CacheImpl(opts) {}

  ComposeFstImplBase(const ComposeFstImplBase &impl) : CacheImpl(impl, true) {
    SetType(impl.Type());
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  virtual ComposeFstImplBase *Copy() const = 0;

  ~ComposeFstImplBase() override {}

  StateId Start() {
    if (!HasStart()) {
      const auto start = ComputeStart();
      if (start != kNoStateId) SetStart(start);
    }
    return CacheImpl::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) SetFinal(s, ComputeFinal(s));
    return CacheImpl::Final(s);
  }

  virtual void Expand(StateId s) = 0;

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl::NumOutputEpsilons(s);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl::InitArcIterator(s, data);
  }

  virtual MatcherBase<Arc> *InitMatcher(const F &fst,
                                        MatchType match_type) const {
    // Use the default matcher if no override is provided.
    return nullptr;
  }

 protected:
  virtual StateId ComputeStart() = 0;
  virtual Weight ComputeFinal(StateId s) = 0;
};

// Implementation of delayed composition templated on the matchers (see
// matcher.h), composition filter (see compose-filter.h) and the composition
// state table (see compose-state-table.h).
template <class CacheStore, class Filter, class StateTable>
class ComposeFstImpl
    : public ComposeFstImplBase<typename CacheStore::Arc, CacheStore> {
 public:
  using Matcher1 = typename Filter::Matcher1;
  using Matcher2 = typename Filter::Matcher2;

  using FST1 = typename Matcher1::FST;
  using FST2 = typename Matcher2::FST;

  using Arc = typename CacheStore::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FilterState = typename Filter::FilterState;
  using State = typename CacheStore::State;

  using CacheImpl = CacheBaseImpl<State, CacheStore>;

  using StateTuple = typename StateTable::StateTuple;

  friend class ComposeFstMatcher<CacheStore, Filter, StateTable>;

  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;
  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;

  template <class M1, class M2>
  ComposeFstImpl(const FST1 &fst1, const FST2 &fst2,
                 const ComposeFstImplOptions<M1, M2, Filter, StateTable,
                                             CacheStore> &opts);

  ComposeFstImpl(const ComposeFstImpl &impl)
      : ComposeFstImplBase<Arc, CacheStore>(impl),
        filter_(new Filter(*impl.filter_, true)),
        matcher1_(filter_->GetMatcher1()),
        matcher2_(filter_->GetMatcher2()),
        fst1_(matcher1_->GetFst()),
        fst2_(matcher2_->GetFst()),
        state_table_(new StateTable(*impl.state_table_)),
        own_state_table_(true),
        match_type_(impl.match_type_) {}

  ~ComposeFstImpl() override {
    if (own_state_table_) delete state_table_;
  }

  ComposeFstImpl *Copy() const override { return new ComposeFstImpl(*this); }

  uint64_t Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64_t Properties(uint64_t mask) const override {
    if ((mask & kError) &&
        (fst1_.Properties(kError, false) || fst2_.Properties(kError, false) ||
         (matcher1_->Properties(0) & kError) ||
         (matcher2_->Properties(0) & kError) |
             (filter_->Properties(0) & kError) ||
         state_table_->Error())) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  // Arranges it so that the first arg to OrderedExpand is the Fst
  // that will be matched on.
  void Expand(StateId s) override {
    const auto &tuple = state_table_->Tuple(s);
    const auto s1 = tuple.StateId1();
    const auto s2 = tuple.StateId2();
    filter_->SetState(s1, s2, tuple.GetFilterState());
    if (MatchInput(s1, s2)) {
      OrderedExpand(s, fst2_, s2, fst1_, s1, matcher2_, true);
    } else {
      OrderedExpand(s, fst1_, s1, fst2_, s2, matcher1_, false);
    }
  }

  const FST1 &GetFst1() const { return fst1_; }

  const FST2 &GetFst2() const { return fst2_; }

  const Matcher1 *GetMatcher1() const { return matcher1_; }

  Matcher1 *GetMatcher1() { return matcher1_; }

  const Matcher2 *GetMatcher2() const { return matcher2_; }

  Matcher2 *GetMatcher2() { return matcher2_; }

  const Filter *GetFilter() const { return filter_.get(); }

  Filter *GetFilter() { return filter_.get(); }

  const StateTable *GetStateTable() const { return state_table_; }

  StateTable *GetStateTable() { return state_table_; }

  MatcherBase<Arc> *InitMatcher(const ComposeFst<Arc, CacheStore> &fst,
                                MatchType match_type) const override {
    const auto test_props = match_type == MATCH_INPUT
                                ? kFstProperties & ~kILabelInvariantProperties
                                : kFstProperties & ~kOLabelInvariantProperties;
    // If both matchers support 'match_type' and we have a guarantee that a
    // call to 'filter_->FilterArc(arc1, arc2)' will not modify the ilabel of
    // arc1 when MATCH_INPUT or the olabel or arc2 when MATCH_OUTPUT, then
    // ComposeFstMatcher can be used.
    if ((matcher1_->Type(false) == match_type) &&
        (matcher2_->Type(false) == match_type) &&
        (filter_->Properties(test_props) == test_props)) {
      return new ComposeFstMatcher<
        CacheStore, Filter, StateTable>(&fst, match_type);
    }
    return nullptr;
  }

 private:
  // This does that actual matching of labels in the composition. The
  // arguments are ordered so matching is called on state 'sa' of
  // 'fsta' for each arc leaving state 'sb' of 'fstb'. The 'match_input' arg
  // determines whether the input or output label of arcs at 'sb' is
  // the one to match on.
  template <class FST, class Matcher>
  void OrderedExpand(StateId s, const Fst<Arc> &, StateId sa, const FST &fstb,
                     StateId sb, Matcher *matchera, bool match_input) {
    matchera->SetState(sa);
    // First processes non-consuming symbols (e.g., epsilons) on FSTA.
    const Arc loop(match_input ? 0 : kNoLabel, match_input ? kNoLabel : 0,
                   Weight::One(), sb);
    MatchArc(s, matchera, loop, match_input);
    // Then processes matches on FSTB.
    for (ArcIterator<FST> iterb(fstb, sb); !iterb.Done(); iterb.Next()) {
      MatchArc(s, matchera, iterb.Value(), match_input);
    }
    CacheImpl::SetArcs(s);
  }

  // Matches a single transition from 'fstb' against 'fata' at 's'.
  template <class Matcher>
  void MatchArc(StateId s, Matcher *matchera, const Arc &arc,
                bool match_input) {
    if (matchera->Find(match_input ? arc.olabel : arc.ilabel)) {
      for (; !matchera->Done(); matchera->Next()) {
        auto arca = matchera->Value();
        auto arcb = arc;
        if (match_input) {
          const auto &fs = filter_->FilterArc(&arcb, &arca);
          if (fs != FilterState::NoState()) AddArc(s, arcb, arca, fs);
        } else {
          const auto &fs = filter_->FilterArc(&arca, &arcb);
          if (fs != FilterState::NoState()) AddArc(s, arca, arcb, fs);
        }
      }
    }
  }

  // Add a matching transition at 's'.
  void AddArc(StateId s, const Arc &arc1, const Arc &arc2,
              const FilterState &f) {
    const StateTuple tuple(arc1.nextstate, arc2.nextstate, f);
    const Arc oarc(arc1.ilabel, arc2.olabel, Times(arc1.weight, arc2.weight),
                   state_table_->FindState(tuple));
    CacheImpl::PushArc(s, oarc);
  }

  StateId ComputeStart() override {
    const auto s1 = fst1_.Start();
    if (s1 == kNoStateId) return kNoStateId;
    const auto s2 = fst2_.Start();
    if (s2 == kNoStateId) return kNoStateId;
    const auto &fs = filter_->Start();
    const StateTuple tuple(s1, s2, fs);
    return state_table_->FindState(tuple);
  }

  Weight ComputeFinal(StateId s) override {
    const auto &tuple = state_table_->Tuple(s);
    const auto s1 = tuple.StateId1();
    auto final1 = matcher1_->Final(s1);
    if (final1 == Weight::Zero()) return final1;
    const auto s2 = tuple.StateId2();
    auto final2 = matcher2_->Final(s2);
    if (final2 == Weight::Zero()) return final2;
    filter_->SetState(s1, s2, tuple.GetFilterState());
    filter_->FilterFinal(&final1, &final2);
    return Times(final1, final2);
  }

  // Determines which side to match on per composition state.
  bool MatchInput(StateId s1, StateId s2) {
    switch (match_type_) {
      case MATCH_INPUT:
        return true;
      case MATCH_OUTPUT:
        return false;
      default:  // MATCH_BOTH
        const auto priority1 = matcher1_->Priority(s1);
        const auto priority2 = matcher2_->Priority(s2);
        if (priority1 == kRequirePriority && priority2 == kRequirePriority) {
          FSTERROR() << "ComposeFst: Both sides can't require match";
          SetProperties(kError, kError);
          return true;
        }
        if (priority1 == kRequirePriority) return false;
        if (priority2 == kRequirePriority) {
          return true;
        }
        return priority1 <= priority2;
    }
  }

  // Identifies and verifies the capabilities of the matcher to be used for
  // composition.
  void SetMatchType();

  std::unique_ptr<Filter> filter_;
  Matcher1 *matcher1_;  // Borrowed reference.
  Matcher2 *matcher2_;  // Borrowed reference.
  const FST1 &fst1_;
  const FST2 &fst2_;
  StateTable *state_table_;
  bool own_state_table_;

  MatchType match_type_;
};

template <class CacheStore, class Filter, class StateTable>
template <class M1, class M2>
ComposeFstImpl<CacheStore, Filter, StateTable>::ComposeFstImpl(
    const FST1 &fst1, const FST2 &fst2,
    const ComposeFstImplOptions<M1, M2, Filter, StateTable, CacheStore> &opts)
    : ComposeFstImplBase<Arc, CacheStore>(opts),
      filter_(opts.filter
                  ? opts.filter
                  : new Filter(fst1, fst2, opts.matcher1, opts.matcher2)),
      matcher1_(filter_->GetMatcher1()),
      matcher2_(filter_->GetMatcher2()),
      fst1_(matcher1_->GetFst()),
      fst2_(matcher2_->GetFst()),
      state_table_(opts.state_table ? opts.state_table
                                    : new StateTable(fst1_, fst2_)),
      own_state_table_(opts.state_table ? opts.own_state_table : true) {
  SetType("compose");
  if (!CompatSymbols(fst2.InputSymbols(), fst1.OutputSymbols())) {
    FSTERROR() << "ComposeFst: Output symbol table of 1st argument "
               << "does not match input symbol table of 2nd argument";
    SetProperties(kError, kError);
  }
  SetInputSymbols(fst1_.InputSymbols());
  SetOutputSymbols(fst2_.OutputSymbols());
  SetMatchType();
  VLOG(2) << "ComposeFstImpl: Match type: " << match_type_;
  if (match_type_ == MATCH_NONE) SetProperties(kError, kError);
  const auto fprops1 = fst1.Properties(kFstProperties, false);
  const auto fprops2 = fst2.Properties(kFstProperties, false);
  const auto mprops1 = matcher1_->Properties(fprops1);
  const auto mprops2 = matcher2_->Properties(fprops2);
  const auto cprops = ComposeProperties(mprops1, mprops2);
  SetProperties(filter_->Properties(cprops), kCopyProperties);
  if (state_table_->Error()) SetProperties(kError, kError);
}

template <class CacheStore, class Filter, class StateTable>
void ComposeFstImpl<CacheStore, Filter, StateTable>::SetMatchType() {
  // Ensures any required matching is possible and known.
  if ((matcher1_->Flags() & kRequireMatch) &&
      matcher1_->Type(true) != MATCH_OUTPUT) {
    FSTERROR() << "ComposeFst: 1st argument cannot perform required matching "
               << "(sort?).";
    match_type_ = MATCH_NONE;
    return;
  }
  if ((matcher2_->Flags() & kRequireMatch) &&
      matcher2_->Type(true) != MATCH_INPUT) {
    FSTERROR() << "ComposeFst: 2nd argument cannot perform required matching "
               << "(sort?).";
    match_type_ = MATCH_NONE;
    return;
  }
  // Finds which sides to match on (favoring minimal testing of capabilities).
  const auto type1 = matcher1_->Type(false);
  const auto type2 = matcher2_->Type(false);
  if (type1 == MATCH_OUTPUT && type2 == MATCH_INPUT) {
    match_type_ = MATCH_BOTH;
  } else if (type1 == MATCH_OUTPUT) {
    match_type_ = MATCH_OUTPUT;
  } else if (type2 == MATCH_INPUT) {
    match_type_ = MATCH_INPUT;
  } else if (matcher1_->Type(true) == MATCH_OUTPUT) {
    match_type_ = MATCH_OUTPUT;
  } else if (matcher2_->Type(true) == MATCH_INPUT) {
    match_type_ = MATCH_INPUT;
  } else {
    FSTERROR() << "ComposeFst: 1st argument cannot match on output labels "
               << "and 2nd argument cannot match on input labels (sort?).";
    match_type_ = MATCH_NONE;
  }
}

}  // namespace internal

// Computes the composition of two transducers. This version is a delayed FST.
// If FST1 transduces string x to y with weight a and FST2 transduces y to z
// with weight b, then their composition transduces string x to z with weight
// Times(x, z).
//
// The output labels of the first transducer or the input labels of the second
// transducer must be sorted (with the default matcher). The weights need to
// form a commutative semiring (valid for TropicalWeight and LogWeight).
//
// Complexity:
//
// Assuming the first FST is unsorted and the second is sorted,
//
//   Time: O(v1 v2 d1 (log d2 + m2)),
//   Space: O(v1 v2)
//
// where vi = # of states visited, di = maximum out-degree, and mi the
// maximum multiplicity of the states visited, for the ith FST. Constant time
// and space to visit an input state or arc is assumed and exclusive of caching.
//
// Caveats:
// - ComposeFst does not trim its output (since it is a delayed operation).
// - The efficiency of composition can be strongly affected by several factors:
//   - the choice of which transducer is sorted - prefer sorting the FST
//     that has the greater average out-degree.
//   - the amount of non-determinism
//   - the presence and location of epsilon transitions - avoid epsilon
//     transitions on the output side of the first transducer or
//     the input side of the second transducer or prefer placing
//     them later in a path since they delay matching and can
//     introduce non-coaccessible states and transitions.
//
// This class attaches interface to implementation and handles reference
// counting, delegating most methods to ImplToFst. The CacheStore specifies the
// cache store (default declared in fst-decl.h).
template <class A, class CacheStore /* = DefaultCacheStore<A> */>
class ComposeFst
    : public ImplToFst<internal::ComposeFstImplBase<A, CacheStore>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = CacheStore;
  using State = typename CacheStore::State;

  using Impl = internal::ComposeFstImplBase<A, CacheStore>;

  friend class ArcIterator<ComposeFst<Arc, CacheStore>>;
  friend class StateIterator<ComposeFst<Arc, CacheStore>>;
  template <class, class, class> friend class ComposeFstMatcher;

  // Compose specifying only caching options.
  ComposeFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
             const CacheOptions &opts = CacheOptions())
      : ImplToFst<Impl>(CreateBase(fst1, fst2, opts)) {}

  // Compose specifying one shared matcher type M. Requires that the input FSTs
  // and matcher FST types be Fst<Arc>. Recommended for best code-sharing and
  // matcher compatiblity.
  template <class Matcher, class Filter, class StateTuple>
  ComposeFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
             const ComposeFstOptions<Arc, Matcher, Filter, StateTuple> &opts)
      : ImplToFst<Impl>(CreateBase1(fst1, fst2, opts)) {}

  // Compose specifying two matcher types Matcher1 and Matcher2. Requires input
  // FST (of the same Arc type, but o.w. arbitrary) match the corresponding
  // matcher FST types). Recommended only for advanced use in demanding or
  // specialized applications due to potential code bloat and matcher
  // incompatibilities.
  template <class Matcher1, class Matcher2, class Filter, class StateTuple>
  ComposeFst(const typename Matcher1::FST &fst1,
             const typename Matcher2::FST &fst2,
             const ComposeFstImplOptions<Matcher1, Matcher2, Filter, StateTuple,
                                         CacheStore> &opts)
      : ImplToFst<Impl>(CreateBase2(fst1, fst2, opts)) {}

  // See Fst<>::Copy() for doc.
  ComposeFst(const ComposeFst<A, CacheStore> &fst, bool safe = false)
      : ImplToFst<Impl>(safe ? std::shared_ptr<Impl>(fst.GetImpl()->Copy())
                             : fst.GetSharedImpl()) {}

  // Get a copy of this ComposeFst. See Fst<>::Copy() for further doc.
  ComposeFst<A, CacheStore> *Copy(bool safe = false) const override {
    return new ComposeFst<A, CacheStore>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  MatcherBase<Arc> *InitMatcher(MatchType match_type) const override {
    return GetImpl()->InitMatcher(*this, match_type);
  }

 protected:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  explicit ComposeFst(std::shared_ptr<Impl> impl) : ImplToFst<Impl>(impl) {}

  // Create compose implementation specifying two matcher types.
  template <class Matcher1, class Matcher2, class Filter, class StateTuple>
  static std::shared_ptr<Impl> CreateBase2(
      const typename Matcher1::FST &fst1, const typename Matcher2::FST &fst2,
      const ComposeFstImplOptions<Matcher1, Matcher2, Filter, StateTuple,
                                  CacheStore> &opts) {
    auto impl = std::make_shared<
        internal::ComposeFstImpl<CacheStore, Filter, StateTuple>>(fst1, fst2,
                                                                  opts);
    if (!(Weight::Properties() & kCommutative) && !opts.allow_noncommute) {
      const auto props1 = fst1.Properties(kUnweighted, true);
      const auto props2 = fst2.Properties(kUnweighted, true);
      if (!(props1 & kUnweighted) && !(props2 & kUnweighted)) {
        FSTERROR() << "ComposeFst: Weights must be a commutative semiring: "
                   << Weight::Type();
        impl->SetProperties(kError, kError);
      }
    }
    return impl;
  }

  // Create compose implementation specifying one matcher type; requires that
  // input and matcher FST types be Fst<Arc>.
  template <class Matcher, class Filter, class StateTuple>
  static std::shared_ptr<Impl> CreateBase1(
      const Fst<Arc> &fst1, const Fst<Arc> &fst2,
      const ComposeFstOptions<Arc, Matcher, Filter, StateTuple> &opts) {
    ComposeFstImplOptions<Matcher, Matcher, Filter, StateTuple, CacheStore>
        nopts(opts, opts.matcher1, opts.matcher2, opts.filter,
              opts.state_table);
    return CreateBase2(fst1, fst2, nopts);
  }

  // Create compose implementation specifying no matcher type.
  static std::shared_ptr<Impl> CreateBase(const Fst<Arc> &fst1,
                                          const Fst<Arc> &fst2,
                                          const CacheOptions &opts) {
    switch (LookAheadMatchType(fst1, fst2)) {  // Check for lookahead matchers
      default:
      case MATCH_NONE: {  // Default composition (no look-ahead).
        ComposeFstOptions<Arc> nopts(opts);
        return CreateBase1(fst1, fst2, nopts);
      }
      case MATCH_OUTPUT: {  // Lookahead on fst1.
        using M = typename DefaultLookAhead<Arc, MATCH_OUTPUT>::FstMatcher;
        using F = typename DefaultLookAhead<Arc, MATCH_OUTPUT>::ComposeFilter;
        ComposeFstOptions<Arc, M, F> nopts(opts);
        return CreateBase1(fst1, fst2, nopts);
      }
      case MATCH_INPUT: {  // Lookahead on fst2
        using M = typename DefaultLookAhead<Arc, MATCH_INPUT>::FstMatcher;
        using F = typename DefaultLookAhead<Arc, MATCH_INPUT>::ComposeFilter;
        ComposeFstOptions<Arc, M, F> nopts(opts);
        return CreateBase1(fst1, fst2, nopts);
      }
    }
  }

 private:
  ComposeFst &operator=(const ComposeFst &fst) = delete;
};

// Specialization for ComposeFst.
template <class Arc, class CacheStore>
class StateIterator<ComposeFst<Arc, CacheStore>>
    : public CacheStateIterator<ComposeFst<Arc, CacheStore>> {
 public:
  explicit StateIterator(const ComposeFst<Arc, CacheStore> &fst)
      : CacheStateIterator<ComposeFst<Arc, CacheStore>>(fst,
                                                        fst.GetMutableImpl()) {}
};

// Specialization for ComposeFst.
template <class Arc, class CacheStore>
class ArcIterator<ComposeFst<Arc, CacheStore>>
    : public CacheArcIterator<ComposeFst<Arc, CacheStore>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const ComposeFst<Arc, CacheStore> &fst, StateId s)
      : CacheArcIterator<ComposeFst<Arc, CacheStore>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class Arc, class CacheStore>
inline void ComposeFst<Arc, CacheStore>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<ComposeFst<Arc, CacheStore>>(*this);
}

// Specialized matcher for ComposeFst. Supports MATCH_INPUT or MATCH_OUTPUT,
// iff the underlying matchers for the two FSTS being composed support
// MATCH_INPUT or MATCH_OUTPUT, respectively.
template <class CacheStore, class Filter, class StateTable>
class ComposeFstMatcher : public MatcherBase<typename CacheStore::Arc> {
 public:
  using Arc = typename CacheStore::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Matcher1 = typename Filter::Matcher1;
  using Matcher2 = typename Filter::Matcher2;
  using FilterState = typename Filter::FilterState;

  using StateTuple = typename StateTable::StateTuple;
  using Impl = internal::ComposeFstImpl<CacheStore, Filter, StateTable>;

  // The compose FST arg must match the filter and state table types.
  // This makes a copy of the FST.
  ComposeFstMatcher(const ComposeFst<Arc, CacheStore> &fst,
                    MatchType match_type)
      : owned_fst_(fst.Copy()),
        fst_(*owned_fst_),
        impl_(static_cast<const Impl *>(fst_.GetImpl())),
        s_(kNoStateId),
        match_type_(match_type),
        matcher1_(impl_->matcher1_->Copy()),
        matcher2_(impl_->matcher2_->Copy()),
        current_loop_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId) {
    if (match_type_ == MATCH_OUTPUT) std::swap(loop_.ilabel, loop_.olabel);
  }

  // The compose FST arg must match the filter and state table types.
  // This doesn't copy the FST (although it may copy components).
  ComposeFstMatcher(const ComposeFst<Arc, CacheStore> *fst,
                    MatchType match_type)
      : fst_(*fst),
        impl_(static_cast<const Impl *>(fst_.GetImpl())),
        s_(kNoStateId),
        match_type_(match_type),
        matcher1_(impl_->matcher1_->Copy()),
        matcher2_(impl_->matcher2_->Copy()),
        current_loop_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId) {
    if (match_type_ == MATCH_OUTPUT) std::swap(loop_.ilabel, loop_.olabel);
  }

  // This makes a copy of the FST.
  ComposeFstMatcher(
      const ComposeFstMatcher<CacheStore, Filter, StateTable> &matcher,
      bool safe = false)
      : owned_fst_(matcher.fst_.Copy(safe)),
        fst_(*owned_fst_),
        impl_(static_cast<const Impl *>(fst_.GetImpl())),
        s_(kNoStateId),
        match_type_(matcher.match_type_),
        matcher1_(matcher.matcher1_->Copy(safe)),
        matcher2_(matcher.matcher2_->Copy(safe)),
        current_loop_(false),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId) {
    if (match_type_ == MATCH_OUTPUT) std::swap(loop_.ilabel, loop_.olabel);
  }

  ComposeFstMatcher<CacheStore, Filter, StateTable> *Copy(
      bool safe = false) const override {
    return new ComposeFstMatcher<CacheStore, Filter, StateTable>(*this, safe);
  }

  MatchType Type(bool test) const override {
    if ((matcher1_->Type(test) == MATCH_NONE) ||
        (matcher2_->Type(test) == MATCH_NONE)) {
      return MATCH_NONE;
    }
    if (((matcher1_->Type(test) == MATCH_UNKNOWN) &&
         (matcher2_->Type(test) == MATCH_UNKNOWN)) ||
        ((matcher1_->Type(test) == MATCH_UNKNOWN) &&
         (matcher2_->Type(test) == match_type_)) ||
        ((matcher1_->Type(test) == match_type_) &&
         (matcher2_->Type(test) == MATCH_UNKNOWN))) {
      return MATCH_UNKNOWN;
    }
    if ((matcher1_->Type(test) == match_type_) &&
        (matcher2_->Type(test) == match_type_)) {
      return match_type_;
    }
    return MATCH_NONE;
  }

  const Fst<Arc> &GetFst() const override { return fst_; }

  uint64_t Properties(uint64_t inprops) const override {
    return inprops;
  }

  void SetState(StateId s) final {
    if (s_ == s) return;
    s_ = s;
    const auto &tuple = impl_->state_table_->Tuple(s);
    matcher1_->SetState(tuple.StateId1());
    matcher2_->SetState(tuple.StateId2());
    loop_.nextstate = s_;
  }

  bool Find(Label label) final {
    bool found = false;
    current_loop_ = false;
    if (label == 0) {
      current_loop_ = true;
      found = true;
    }
    if (match_type_ == MATCH_INPUT) {
      found = found || FindLabel(label, matcher1_.get(), matcher2_.get());
    } else {  // match_type_ == MATCH_OUTPUT
      found = found || FindLabel(label, matcher2_.get(), matcher1_.get());
    }
    return found;
  }

  bool Done() const final {
    return !current_loop_ && matcher1_->Done() && matcher2_->Done();
  }

  const Arc &Value() const final { return current_loop_ ? loop_ : arc_; }

  void Next() final {
    if (current_loop_) {
      current_loop_ = false;
    } else if (match_type_ == MATCH_INPUT) {
      FindNext(matcher1_.get(), matcher2_.get());
    } else {  // match_type_ == MATCH_OUTPUT
      FindNext(matcher2_.get(), matcher1_.get());
    }
  }

  std::ptrdiff_t Priority(StateId s) final { return fst_.NumArcs(s); }

 private:
  // Processes a match with the filter and creates resulting arc.
  bool MatchArc(StateId s, Arc arc1,
                Arc arc2) {  // FIXME(kbg): copy but not assignment.
    const auto &fs = impl_->filter_->FilterArc(&arc1, &arc2);
    if (fs == FilterState::NoState()) return false;
    const StateTuple tuple(arc1.nextstate, arc2.nextstate, fs);
    arc_.ilabel = arc1.ilabel;
    arc_.olabel = arc2.olabel;
    arc_.weight = Times(arc1.weight, arc2.weight);
    arc_.nextstate = impl_->state_table_->FindState(tuple);
    return true;
  }

  // Finds the first match allowed by the filter.
  template <class MatcherA, class MatcherB>
  bool FindLabel(Label label, MatcherA *matchera, MatcherB *matcherb) {
    if (matchera->Find(label)) {
      matcherb->Find(match_type_ == MATCH_INPUT ? matchera->Value().olabel
                                                : matchera->Value().ilabel);
      return FindNext(matchera, matcherb);
    }
    return false;
  }

  // Finds the next match allowed by the filter, returning true iff such a
  // match is found.
  template <class MatcherA, class MatcherB>
  bool FindNext(MatcherA *matchera, MatcherB *matcherb) {
    // State when entering this function:
    // 'matchera' is pointed to a match x, y for label x, and a match for y was
    // requested on 'matcherb'.
    while (!matchera->Done() || !matcherb->Done()) {
      if (matcherb->Done()) {
        // If no more matches for y on 'matcherb', moves forward on 'matchera'
        // until a match x, y' is found such that there is a match for y' on
        // 'matcherb'.
        matchera->Next();
        while (!matchera->Done() &&
               !matcherb->Find(match_type_ == MATCH_INPUT
                                   ? matchera->Value().olabel
                                   : matchera->Value().ilabel)) {
          matchera->Next();
        }
      }
      while (!matcherb->Done()) {
        // 'matchera' is pointing to a match x, y' ('arca') and 'matcherb' is
        // pointing to a match y', z' ('arcb'). If combining these two arcs is
        // allowed by the filter (hence resulting in an arc x, z') return true.
        // Position 'matcherb' on the next potential match for y' before
        // returning.
        const auto &arca = matchera->Value();
        const auto &arcb = matcherb->Value();
        // Position 'matcherb' on the next potential match for y'.
        matcherb->Next();
        // Returns true If combining these two arcs is allowed by the filter
        // (hence resulting in an arc x, z'); otherwise consider next match
        // for y' on 'matcherb'.
        if (MatchArc(s_, match_type_ == MATCH_INPUT ? arca : arcb,
                     match_type_ == MATCH_INPUT ? arcb : arca)) {
          return true;
        }
      }
    }
    // Both 'matchera' and 'matcherb' are done, no more match to analyse.
    return false;
  }

  std::unique_ptr<const ComposeFst<Arc, CacheStore>> owned_fst_;
  const ComposeFst<Arc, CacheStore> &fst_;
  const Impl *impl_;
  StateId s_;
  MatchType match_type_;
  std::unique_ptr<Matcher1> matcher1_;
  std::unique_ptr<Matcher2> matcher2_;
  bool current_loop_;
  Arc loop_;
  Arc arc_;
};

// Useful alias when using StdArc.
using StdComposeFst = ComposeFst<StdArc>;

enum ComposeFilter {
  AUTO_FILTER,
  NULL_FILTER,
  TRIVIAL_FILTER,
  SEQUENCE_FILTER,
  ALT_SEQUENCE_FILTER,
  MATCH_FILTER
};

struct ComposeOptions {
  bool connect;               // Connect output?
  ComposeFilter filter_type;  // Pre-defined filter to use.

  explicit ComposeOptions(bool connect = true,
                          ComposeFilter filter_type = AUTO_FILTER)
      : connect(connect), filter_type(filter_type) {}
};

// Computes the composition of two transducers. This version writes
// the composed FST into a MutableFst. If FST1 transduces string x to
// y with weight a and FST2 transduces y to z with weight b, then
// their composition transduces string x to z with weight
// Times(x, z).
//
// The output labels of the first transducer or the input labels of
// the second transducer must be sorted.  The weights need to form a
// commutative semiring (valid for TropicalWeight and LogWeight).
//
// Complexity:
//
// Assuming the first FST is unsorted and the second is sorted:
//
//   Time: O(V1 V2 D1 (log D2 + M2)),
//   Space: O(V1 V2 D1 M2)
//
// where Vi = # of states, Di = maximum out-degree, and Mi is the maximum
// multiplicity, for the ith FST.
//
// Caveats:
//
// - Compose trims its output.
// - The efficiency of composition can be strongly affected by several factors:
//   - the choice of which transducer is sorted - prefer sorting the FST
//     that has the greater average out-degree.
//   - the amount of non-determinism
//   - the presence and location of epsilon transitions - avoid epsilon
//     transitions on the output side of the first transducer or
//     the input side of the second transducer or prefer placing
//     them later in a path since they delay matching and can
//     introduce non-coaccessible states and transitions.
template <class Arc>
void Compose(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
             MutableFst<Arc> *ofst,
             const ComposeOptions &opts = ComposeOptions()) {
  using M = Matcher<Fst<Arc>>;
  // In each case, we cache only the last state for fastest copy.
  switch (opts.filter_type) {
    case AUTO_FILTER: {
      CacheOptions nopts;
      nopts.gc_limit = 0;
      *ofst = ComposeFst<Arc>(ifst1, ifst2, nopts);
      break;
    }
    case NULL_FILTER: {
      ComposeFstOptions<Arc, M, NullComposeFilter<M>> copts;
      copts.gc_limit = 0;
      *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
      break;
    }
    case SEQUENCE_FILTER: {
      ComposeFstOptions<Arc, M, SequenceComposeFilter<M>> copts;
      copts.gc_limit = 0;
      *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
      break;
    }
    case ALT_SEQUENCE_FILTER: {
      ComposeFstOptions<Arc, M, AltSequenceComposeFilter<M>> copts;
      copts.gc_limit = 0;
      *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
      break;
    }
    case MATCH_FILTER: {
      ComposeFstOptions<Arc, M, MatchComposeFilter<M>> copts;
      copts.gc_limit = 0;
      *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
      break;
    }
    case TRIVIAL_FILTER: {
      ComposeFstOptions<Arc, M, TrivialComposeFilter<M>> copts;
      copts.gc_limit = 0;
      *ofst = ComposeFst<Arc>(ifst1, ifst2, copts);
      break;
    }
  }
  if (opts.connect) Connect(ofst);
}

}  // namespace fst

#endif  // FST_COMPOSE_H_
