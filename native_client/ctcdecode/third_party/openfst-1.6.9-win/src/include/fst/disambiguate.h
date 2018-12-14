// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to disambiguate an FST.

#ifndef FST_DISAMBIGUATE_H_
#define FST_DISAMBIGUATE_H_

#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/connect.h>
#include <fst/determinize.h>
#include <fst/dfs-visit.h>
#include <fst/project.h>
#include <fst/prune.h>
#include <fst/state-map.h>
#include <fst/state-table.h>
#include <fst/union-find.h>
#include <fst/verify.h>


namespace fst {

template <class Arc>
struct DisambiguateOptions : public DeterminizeOptions<Arc> {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit DisambiguateOptions(float delta = kDelta,
                               Weight weight = Weight::Zero(),
                               StateId n = kNoStateId, Label label = 0)
      : DeterminizeOptions<Arc>(delta, std::move(weight), n, label,
                                DETERMINIZE_FUNCTIONAL) {}
};

namespace internal {

// A determinization filter based on a subset element relation. The relation is
// assumed to be reflexive and symmetric.
template <class Arc, class Relation>
class RelationDeterminizeFilter {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FilterState = IntegerFilterState<StateId>;
  using StateTuple = DeterminizeStateTuple<Arc, FilterState>;
  using Subset = typename StateTuple::Subset;
  using Element = typename StateTuple::Element;
  using LabelMap = std::multimap<Label, DeterminizeArc<StateTuple>>;

  // This is needed (e.g.) to go into the gallic domain for transducers; there
  // is no need to rebind the relation since its use here only depends on the
  // state IDs.
  template <class A>
  struct rebind {
    using Other = RelationDeterminizeFilter<A, Relation>;
  };

  explicit RelationDeterminizeFilter(const Fst<Arc> &fst)
      : fst_(fst.Copy()), r_(new Relation()), s_(kNoStateId), head_(nullptr) {}

  // Ownership of the relation is given to this class.
  RelationDeterminizeFilter(const Fst<Arc> &fst, Relation *r)
      : fst_(fst.Copy()), r_(r), s_(kNoStateId), head_(0) {}

  // Ownership of the relation is given to this class.
  RelationDeterminizeFilter(const Fst<Arc> &fst, Relation *r,
                            std::vector<StateId> *head)
      : fst_(fst.Copy()), r_(r), s_(kNoStateId), head_(head) {}

  // This is needed, e.g., to go into the gallic domain for transducers.
  // Ownership of the templated filter argument is given to this class.
  template <class Filter>
  RelationDeterminizeFilter(const Fst<Arc> &fst, Filter *filter)
      : fst_(fst.Copy()),
        r_(new Relation(filter->GetRelation())),
        s_(kNoStateId),
        head_(filter->GetHeadStates()) {
    delete filter;
  }

  // Copy constructor; the FST can be passed if it has been deep-copied.
  RelationDeterminizeFilter(const RelationDeterminizeFilter &filter,
                            const Fst<Arc> *fst = nullptr)
      : fst_(fst ? fst->Copy() : filter.fst_->Copy()),
        r_(new Relation(*filter.r_)),
        s_(kNoStateId),
        head_() {}

  FilterState Start() const { return FilterState(fst_->Start()); }

  void SetState(StateId s, const StateTuple &tuple) {
    if (s_ != s) {
      s_ = s;
      tuple_ = &tuple;
      const auto head = tuple.filter_state.GetState();
      is_final_ = fst_->Final(head) != Weight::Zero();
      if (head_) {
        if (head_->size() <= s) head_->resize(s + 1, kNoStateId);
        (*head_)[s] = head;
      }
    }
  }

  // Filters transition, possibly modifying label map. Returns true if arc is
  // added to label map.
  bool FilterArc(const Arc &arc, const Element &src_element,
                 const Element &dest_element, LabelMap *label_map) const;

  // Filters super-final transition, returning new final weight.
  Weight FilterFinal(const Weight final_weight, const Element &element) const {
    return is_final_ ? final_weight : Weight::Zero();
  }

  static uint64_t Properties(uint64_t props) {
    return props & ~(kIDeterministic | kODeterministic);
  }

  const Relation &GetRelation() { return *r_; }

  std::vector<StateId> *GetHeadStates() { return head_; }

 private:
  // Pairs arc labels with state tuples with possible heads and empty subsets.
  void InitLabelMap(LabelMap *label_map) const;

  std::unique_ptr<Fst<Arc>> fst_;  // Input FST.
  std::unique_ptr<Relation> r_;    // Relation compatible with inv. trans. fnc.
  StateId s_;                      // Current state.
  const StateTuple *tuple_;        // Current tuple.
  bool is_final_;                  // Is the current head state final?
  std::vector<StateId> *head_;     // Head state for a given state,
                                   // owned by the Disambiguator.
};

template <class Arc, class Relation>
bool RelationDeterminizeFilter<Arc, Relation>::FilterArc(
    const Arc &arc, const Element &src_element, const Element &dest_element,
    LabelMap *label_map) const {
  bool added = false;
  if (label_map->empty()) InitLabelMap(label_map);
  // Adds element to state tuple if element state is related to tuple head.
  for (auto liter = label_map->lower_bound(arc.ilabel);
       liter != label_map->end() && liter->first == arc.ilabel; ++liter) {
    auto *dest_tuple = liter->second.dest_tuple;
    const auto dest_head = dest_tuple->filter_state.GetState();
    if ((*r_)(dest_element.state_id, dest_head)) {
      dest_tuple->subset.push_front(dest_element);
      added = true;
    }
  }
  return added;
}

template <class Arc, class Relation>
void RelationDeterminizeFilter<Arc, Relation>::InitLabelMap(
    LabelMap *label_map) const {
  const auto src_head = tuple_->filter_state.GetState();
  Label label = kNoLabel;
  StateId nextstate = kNoStateId;
  for (ArcIterator<Fst<Arc>> aiter(*fst_, src_head); !aiter.Done();
       aiter.Next()) {
    const auto &arc = aiter.Value();
    // Continues if multiarc.
    if (arc.ilabel == label && arc.nextstate == nextstate) continue;
    DeterminizeArc<StateTuple> det_arc(arc);
    det_arc.dest_tuple->filter_state = FilterState(arc.nextstate);
    label_map->insert(std::make_pair(arc.ilabel, det_arc));
    label = arc.ilabel;
    nextstate = arc.nextstate;
  }
}

// Helper class to disambiguate an FST via Disambiguate().
template <class Arc>
class Disambiguator {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // IDs arcs with state ID and arc position. Arc position -1 indicates final
  // (super-final transition).
  using ArcId = std::pair<StateId, std::ptrdiff_t>;

  Disambiguator() : error_(false) {}

  void Disambiguate(
      const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
      const DisambiguateOptions<Arc> &opts = DisambiguateOptions<Arc>()) {
    VectorFst<Arc> sfst(ifst);
    Connect(&sfst);
    ArcSort(&sfst, ArcCompare());
    PreDisambiguate(sfst, ofst, opts);
    ArcSort(ofst, ArcCompare());
    FindAmbiguities(*ofst);
    RemoveSplits(ofst);
    MarkAmbiguities();
    RemoveAmbiguities(ofst);
    if (error_) ofst->SetProperties(kError, kError);
  }

 private:
  // Comparison functor for comparing input labels and next states of arcs. This
  // sort order facilitates the predisambiguation.
  class ArcCompare {
   public:
    bool operator()(const Arc &arc1, const Arc &arc2) const {
      return arc1.ilabel < arc2.ilabel ||
             (arc1.ilabel == arc2.ilabel && arc1.nextstate < arc2.nextstate);
    }

    uint64_t Properties(uint64_t props) const {
      return (props & kArcSortProperties) | kILabelSorted |
             (props & kAcceptor ? kOLabelSorted : 0);
    }
  };

  // Comparison functor for comparing transitions represented by their arc ID.
  // This sort order facilitates ambiguity detection.
  class ArcIdCompare {
   public:
    explicit ArcIdCompare(const std::vector<StateId> &head) : head_(head) {}

    bool operator()(const ArcId &a1, const ArcId &a2) const {
      // Sort first by source head state...
      const auto src1 = a1.first;
      const auto src2 = a2.first;
      const auto head1 = head_[src1];
      const auto head2 = head_[src2];
      if (head1 < head2) return true;
      if (head2 < head1) return false;
      // ...then by source state...
      if (src1 < src2) return true;
      if (src2 < src1) return false;
      // ...then by position.
      return a1.second < a2.second;
    }

   private:
    const std::vector<StateId> &head_;
  };

  // A relation that determines if two states share a common future.
  class CommonFuture {
   public:
    using StateTable = GenericComposeStateTable<Arc, TrivialFilterState>;
    using StateTuple = typename StateTable::StateTuple;

    // Needed for compilation with DeterminizeRelationFilter.
    CommonFuture() {
      FSTERROR() << "Disambiguate::CommonFuture: FST not provided";
    }

    explicit CommonFuture(const Fst<Arc> &ifst) {
      using M = Matcher<Fst<Arc>>;
      ComposeFstOptions<Arc, M, NullComposeFilter<M>> opts;
      // Ensures composition is between acceptors.
      const bool trans = ifst.Properties(kNotAcceptor, true);
      const auto *fsa =
          trans ? new ProjectFst<Arc>(ifst, PROJECT_INPUT) : &ifst;
      opts.state_table = new StateTable(*fsa, *fsa);
      const ComposeFst<Arc> cfst(*fsa, *fsa, opts);
      std::vector<bool> coaccess;
      uint64_t props = 0;
      SccVisitor<Arc> scc_visitor(nullptr, nullptr, &coaccess, &props);
      DfsVisit(cfst, &scc_visitor);
      for (StateId s = 0; s < coaccess.size(); ++s) {
        if (coaccess[s]) {
          related_.insert(opts.state_table->Tuple(s).StatePair());
        }
      }
      if (trans) delete fsa;
    }

    bool operator()(const StateId s1, StateId s2) const {
      return related_.count(std::make_pair(s1, s2)) > 0;
    }

   private:
    // States s1 and s2 resp. are in this relation iff they there is a
    // path from s1 to a final state that has the same label as some
    // path from s2 to a final state.
    std::set<std::pair<StateId, StateId>> related_;
  };

  using ArcIdMap = std::multimap<ArcId, ArcId, ArcIdCompare>;

  // Inserts candidate into the arc ID map.
  inline void InsertCandidate(StateId s1, StateId s2, const ArcId &a1,
                              const ArcId &a2) {
    candidates_->insert(head_[s1] > head_[s2] ? std::make_pair(a1, a2)
                                              : std::make_pair(a2, a1));
  }

  // Returns the arc corresponding to ArcId a.
  static Arc GetArc(const Fst<Arc> &fst, ArcId aid) {
    if (aid.second == -1) {  // Returns super-final transition.
      return Arc(kNoLabel, kNoLabel, fst.Final(aid.first), kNoStateId);
    } else {
      ArcIterator<Fst<Arc>> aiter(fst, aid.first);
      aiter.Seek(aid.second);
      return aiter.Value();
    }
  }

  // Outputs an equivalent FST whose states are subsets of states that have a
  // future path in common.
  void PreDisambiguate(const ExpandedFst<Arc> &ifst, MutableFst<Arc> *ofst,
                       const DisambiguateOptions<Arc> &opts);

  // Finds transitions that are ambiguous candidates in the result of
  // PreDisambiguate.
  void FindAmbiguities(const ExpandedFst<Arc> &fst);

  // Finds transition pairs that are ambiguous candidates from two specified
  // source states.
  void FindAmbiguousPairs(const ExpandedFst<Arc> &fst, StateId s1, StateId s2);

  // Marks ambiguous transitions to be removed.
  void MarkAmbiguities();

  // Deletes spurious ambiguous transitions (due to quantization).
  void RemoveSplits(MutableFst<Arc> *ofst);

  // Deletes actual ambiguous transitions.
  void RemoveAmbiguities(MutableFst<Arc> *ofst);

  // States s1 and s2 are in this relation iff there is a path from the initial
  // state to s1 that has the same label as some path from the initial state to
  // s2. We store only state pairs s1, s2 such that s1 <= s2.
  std::set<std::pair<StateId, StateId>> coreachable_;

  // Queue of disambiguation-related states to be processed. We store only
  // state pairs s1, s2 such that s1 <= s2.
  std::list<std::pair<StateId, StateId>> queue_;

  // Head state in the pre-disambiguation for a given state.
  std::vector<StateId> head_;

  // Maps from a candidate ambiguous arc A to each ambiguous candidate arc B
  // with the same label and destination state as A, whose source state s' is
  // coreachable with the source state s of A, and for which head(s') < head(s).
  std::unique_ptr<ArcIdMap> candidates_;

  // Set of ambiguous transitions to be removed.
  std::set<ArcId> ambiguous_;

  // States to merge due to quantization issues.
  std::unique_ptr<UnionFind<StateId>> merge_;
  // Marks error condition.
  bool error_;

  Disambiguator(const Disambiguator &) = delete;
  Disambiguator &operator=(const Disambiguator &) = delete;
};

template <class Arc>
void Disambiguator<Arc>::PreDisambiguate(const ExpandedFst<Arc> &ifst,
                                         MutableFst<Arc> *ofst,
                                         const DisambiguateOptions<Arc> &opts) {
  using CommonDivisor = DefaultCommonDivisor<Weight>;
  using Filter = RelationDeterminizeFilter<Arc, CommonFuture>;
  // Subset elements with states s1 and s2 (resp.) are in this relation iff they
  // there is a path from s1 to a final state that has the same label as some
  // path from s2 to a final state.
  auto *common_future = new CommonFuture(ifst);
  DeterminizeFstOptions<Arc, CommonDivisor, Filter> nopts;
  nopts.delta = opts.delta;
  nopts.subsequential_label = opts.subsequential_label;
  nopts.filter = new Filter(ifst, common_future, &head_);
  // The filter takes ownership of 'common_future', and determinization takes
  // ownership of the filter itself.
  nopts.gc_limit = 0;  // Cache only the last state for fastest copy.
  if (opts.weight_threshold != Weight::Zero() ||
      opts.state_threshold != kNoStateId) {
    /* TODO(riley): fails regression test; understand why
    if (ifst.Properties(kAcceptor, true)) {
      std::vector<Weight> idistance, odistance;
      ShortestDistance(ifst, &idistance, true);
      DeterminizeFst<Arc> dfst(ifst, &idistance, &odistance, nopts);
      PruneOptions< Arc, AnyArcFilter<Arc>> popts(opts.weight_threshold,
                                                   opts.state_threshold,
                                                   AnyArcFilter<Arc>(),
                                                   &odistance);
      Prune(dfst, ofst, popts);
      } else */ {
      *ofst = DeterminizeFst<Arc>(ifst, nopts);
      Prune(ofst, opts.weight_threshold, opts.state_threshold);
    }
  } else {
    *ofst = DeterminizeFst<Arc>(ifst, nopts);
  }
  head_.resize(ofst->NumStates(), kNoStateId);
}

template <class Arc>
void Disambiguator<Arc>::FindAmbiguities(const ExpandedFst<Arc> &fst) {
  if (fst.Start() == kNoStateId) return;
  candidates_.reset(new ArcIdMap(ArcIdCompare(head_)));
  const auto start_pr = std::make_pair(fst.Start(), fst.Start());
  coreachable_.insert(start_pr);
  queue_.push_back(start_pr);
  while (!queue_.empty()) {
    const auto &pr = queue_.front();
    const auto s1 = pr.first;
    const auto s2 = pr.second;
    queue_.pop_front();
    FindAmbiguousPairs(fst, s1, s2);
  }
}

template <class Arc>
void Disambiguator<Arc>::FindAmbiguousPairs(const ExpandedFst<Arc> &fst,
                                            StateId s1, StateId s2) {
  if (fst.NumArcs(s2) > fst.NumArcs(s1)) FindAmbiguousPairs(fst, s2, s1);
  SortedMatcher<Fst<Arc>> matcher(fst, MATCH_INPUT);
  matcher.SetState(s2);
  for (ArcIterator<Fst<Arc>> aiter(fst, s1); !aiter.Done(); aiter.Next()) {
    const auto &arc1 = aiter.Value();
    const ArcId a1(s1, aiter.Position());
    if (matcher.Find(arc1.ilabel)) {
      for (; !matcher.Done(); matcher.Next()) {
        const auto &arc2 = matcher.Value();
        // Continues on implicit epsilon match.
        if (arc2.ilabel == kNoLabel) continue;
        const ArcId a2(s2, matcher.Position());
        // Actual transition is ambiguous.
        if (s1 != s2 && arc1.nextstate == arc2.nextstate) {
          InsertCandidate(s1, s2, a1, a2);
        }
        const auto spr = arc1.nextstate <= arc2.nextstate
                             ? std::make_pair(arc1.nextstate, arc2.nextstate)
                             : std::make_pair(arc2.nextstate, arc1.nextstate);
        // Not already marked as coreachable?
        if (coreachable_.insert(spr).second) {
          // Only possible if state split by quantization issues.
          if (spr.first != spr.second &&
              head_[spr.first] == head_[spr.second]) {
            if (!merge_) {
              merge_.reset(new UnionFind<StateId>(fst.NumStates(), kNoStateId));
              merge_->MakeAllSet(fst.NumStates());
            }
            merge_->Union(spr.first, spr.second);
          } else {
            queue_.push_back(spr);
          }
        }
      }
    }
  }
  // Super-final transition is ambiguous.
  if (s1 != s2 && fst.Final(s1) != Weight::Zero() &&
      fst.Final(s2) != Weight::Zero()) {
    const ArcId a1(s1, -1);
    const ArcId a2(s2, -1);
    InsertCandidate(s1, s2, a1, a2);
  }
}

template <class Arc>
void Disambiguator<Arc>::MarkAmbiguities() {
  if (!candidates_) return;
  for (auto it = candidates_->begin(); it != candidates_->end(); ++it) {
    const auto a = it->first;
    const auto b = it->second;
    // If b is not to be removed, then a is.
    if (ambiguous_.count(b) == 0) ambiguous_.insert(a);
  }
  coreachable_.clear();
  candidates_.reset();
}

template <class Arc>
void Disambiguator<Arc>::RemoveSplits(MutableFst<Arc> *ofst) {
  if (!merge_) return;
  // Merges split states to remove spurious ambiguities.
  for (StateIterator<MutableFst<Arc>> siter(*ofst); !siter.Done();
       siter.Next()) {
    for (MutableArcIterator<MutableFst<Arc>> aiter(ofst, siter.Value());
         !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      const auto nextstate = merge_->FindSet(arc.nextstate);
      if (nextstate != arc.nextstate) {
        arc.nextstate = nextstate;
        aiter.SetValue(arc);
      }
    }
  }
  // Repeats search for actual ambiguities on modified FST.
  coreachable_.clear();
  merge_.reset();
  candidates_.reset();
  FindAmbiguities(*ofst);
  if (merge_) {  // Shouldn't get here; sanity test.
    FSTERROR() << "Disambiguate: Unable to remove spurious ambiguities";
    error_ = true;
    return;
  }
}

template <class Arc>
void Disambiguator<Arc>::RemoveAmbiguities(MutableFst<Arc> *ofst) {
  if (ambiguous_.empty()) return;
  // Adds dead state to redirect ambiguous transitions to be removed.
  const auto dead = ofst->AddState();
  for (auto it = ambiguous_.begin(); it != ambiguous_.end(); ++it) {
    const auto pos = it->second;
    if (pos >= 0) {  // Actual transition.
      MutableArcIterator<MutableFst<Arc>> aiter(ofst, it->first);
      aiter.Seek(pos);
      auto arc = aiter.Value();
      arc.nextstate = dead;
      aiter.SetValue(arc);
    } else {  // Super-final transition.
      ofst->SetFinal(it->first, Weight::Zero());
    }
  }
  Connect(ofst);
  ambiguous_.clear();
}

}  // namespace internal

// Disambiguates a weighted FST. This version writes the disambiguated FST to an
// output MutableFst. The result will be an equivalent FST that has the
// property that there are not two distinct paths from the initial state to a
// final state with the same input labeling.
//
// The weights must be (weakly) left divisible (valid for Tropical and
// LogWeight).
//
// Complexity:
//
//   Disambiguable: exponential (polynomial in the size of the output).
//   Non-disambiguable: does not terminate.
//
// The disambiguable transducers include all automata and functional transducers
// that are unweighted or that are acyclic or that are unambiguous.
//
// For more information, see:
//
// Mohri, M. and Riley, M. 2015. On the disambiguation of weighted automata.
// In CIAA, pages 263-278.
template <class Arc>
void Disambiguate(
    const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
    const DisambiguateOptions<Arc> &opts = DisambiguateOptions<Arc>()) {
  internal::Disambiguator<Arc> disambiguator;
  disambiguator.Disambiguate(ifst, ofst, opts);
}

}  // namespace fst

#endif  // FST_DISAMBIGUATE_H_
