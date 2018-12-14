// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes to add lookahead to FST matchers, useful for improving composition
// efficiency with certain inputs.

#ifndef FST_LOOKAHEAD_MATCHER_H_
#define FST_LOOKAHEAD_MATCHER_H_

#include <memory>
#include <utility>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/add-on.h>
#include <fst/const-fst.h>
#include <fst/fst.h>
#include <fst/label-reachable.h>
#include <fst/matcher.h>


DECLARE_string(save_relabel_ipairs);
DECLARE_string(save_relabel_opairs);

namespace fst {

// Lookahead matches extend the matcher interface with following additional
// methods:
//
// template <class FST>
// class LookAheadMatcher {
//  public:
//   using Arc = typename FST::Arc;
//   using Label = typename Arc::Label;
//   using StateId = typename Arc::StateId;
//   using Weight = typename Arc::Weight;
//
//  // Required constructors.
//  // This makes a copy of the FST.
//  LookAheadMatcher(const FST &fst, MatchType match_type);
//  // This doesn't copy the FST.
//  LookAheadMatcher(const FST *fst, MatchType match_type);
//  // This makes a copy of the FST.
//  // See Copy() below.
//  LookAheadMatcher(const LookAheadMatcher &matcher, bool safe = false);
//
//   // If safe = true, the copy is thread-safe (except the lookahead FST is
//   // preserved). See Fst<>::Copy() for further doc.
//   LookaheadMatcher<FST> *Copy(bool safe = false) const override;

//  // Below are methods for looking ahead for a match to a label and more
//  // generally, to a rational set. Each returns false if there is definitely
//  // not a match and returns true if there possibly is a match.
//
//  // Optionally pre-specifies the lookahead FST that will be passed to
//  // LookAheadFst() for possible precomputation. If copy is true, then the FST
//  // argument is a copy of the FST used in the previous call to this method
//  // (to avoid unnecessary updates).
//  void InitLookAheadFst(const Fst<Arc> &fst, bool copy = false) override;
//
//  // Are there paths from a state in the lookahead FST that can be read from
//  // the curent matcher state?
//  bool LookAheadFst(const Fst<Arc> &fst, StateId s) override;
//
//  // Can the label be read from the current matcher state after possibly
//  // following epsilon transitions?
//  bool LookAheadLabel(Label label) const override;
//
//  // The following methods allow looking ahead for an arbitrary rational set
//  // of strings, specified by an FST and a state from which to begin the
//  // matching. If the lookahead FST is a transducer, this looks on the side
//  // different from the matcher's match_type (cf. composition).
//  // Is there is a single non-epsilon arc found in the lookahead FST that
//  // begins the path (after possibly following any epsilons) in the last call
//  // to LookAheadFst? If so, return true and copy it to the arc argument;
//  // otherwise, return false. Non-trivial implementations are useful for
//  // label-pushing in composition.
//  bool LookAheadPrefix(Arc *arc) override;
//
//  // Gives an estimate of the combined weight of the paths in the lookahead
//  // and matcher FSTs for the last call to LookAheadFst. Non-trivial
//  // implementations are useful for weight-pushing in composition.
//  Weight LookAheadWeight() const override;
// };

// Look-ahead flags.
// Matcher is a lookahead matcher when match_type is MATCH_INPUT.
constexpr uint32_t kInputLookAheadMatcher = 0x00000010;

// Matcher is a lookahead matcher when match_type is MATCH_OUTPUT.
constexpr uint32_t kOutputLookAheadMatcher = 0x00000020;

// Is a non-trivial implementation of LookAheadWeight() method defined and
// if so, should it be used?
constexpr uint32_t kLookAheadWeight = 0x00000040;

// Is a non-trivial implementation of LookAheadPrefix() method defined and
// if so, should it be used?
constexpr uint32_t kLookAheadPrefix = 0x00000080;

// Look-ahead of matcher FST non-epsilon arcs?
constexpr uint32_t kLookAheadNonEpsilons = 0x00000100;

// Look-ahead of matcher FST epsilon arcs?
constexpr uint32_t kLookAheadEpsilons = 0x00000200;

// Ignore epsilon paths for the lookahead prefix? This gives correct results in
// composition only with an appropriate composition filter since it depends on
// the filter blocking the ignored paths.
constexpr uint32_t kLookAheadNonEpsilonPrefix = 0x00000400;

// For LabelLookAheadMatcher, save relabeling data to file?
constexpr uint32_t kLookAheadKeepRelabelData = 0x00000800;

// Flags used for lookahead matchers.
constexpr uint32_t kLookAheadFlags = 0x00000ff0;

// LookAhead Matcher interface, templated on the Arc definition; used
// for lookahead matcher specializations that are returned by the
// InitMatcher() Fst method.
template <class Arc>
class LookAheadMatcherBase : public MatcherBase<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  virtual void InitLookAheadFst(const Fst<Arc> &, bool copy = false) = 0;
  virtual bool LookAheadFst(const Fst<Arc> &, StateId) = 0;
  virtual bool LookAheadLabel(Label) const = 0;

  // Suggested concrete implementation of lookahead methods.

  bool LookAheadPrefix(Arc *arc) const {
    if (prefix_arc_.nextstate != kNoStateId) {
      *arc = prefix_arc_;
      return true;
    } else {
      return false;
    }
  }

  Weight LookAheadWeight() const { return weight_; }

 protected:
  // Concrete implementations for lookahead helper methods.

  void ClearLookAheadWeight() { weight_ = Weight::One(); }

  void SetLookAheadWeight(Weight weight) { weight_ = std::move(weight); }

  void ClearLookAheadPrefix() { prefix_arc_.nextstate = kNoStateId; }

  void SetLookAheadPrefix(Arc arc) { prefix_arc_ = std::move(arc); }

 private:
  Arc prefix_arc_;
  Weight weight_;
};

// Doesn't actually lookahead, just declares that the future looks good.
template <class M>
class TrivialLookAheadMatcher
    : public LookAheadMatcherBase<typename M::FST::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST.
  TrivialLookAheadMatcher(const FST &fst, MatchType match_type)
      : matcher_(fst, match_type) {}

  // This doesn't copy the FST.
  TrivialLookAheadMatcher(const FST *fst, MatchType match_type)
      : matcher_(fst, match_type) {}

  // This makes a copy of the FST.
  TrivialLookAheadMatcher(const TrivialLookAheadMatcher<M> &lmatcher,
                          bool safe = false)
      : matcher_(lmatcher.matcher_, safe) {}

  TrivialLookAheadMatcher<M> *Copy(bool safe = false) const override {
    return new TrivialLookAheadMatcher<M>(*this, safe);
  }

  MatchType Type(bool test) const override { return matcher_.Type(test); }

  void SetState(StateId s) final { return matcher_.SetState(s); }

  bool Find(Label label) final { return matcher_.Find(label); }

  bool Done() const final { return matcher_.Done(); }

  const Arc &Value() const final { return matcher_.Value(); }

  void Next() final { matcher_.Next(); }

  Weight Final(StateId s) const final { return matcher_.Final(s); }

  std::ptrdiff_t Priority(StateId s) final { return matcher_.Priority(s); }

  const FST &GetFst() const override { return matcher_.GetFst(); }

  uint64_t Properties(uint64_t props) const override {
    return matcher_.Properties(props);
  }

  uint32_t Flags() const override {
    return matcher_.Flags() | kInputLookAheadMatcher | kOutputLookAheadMatcher;
  }

  // Lookahead methods (all trivial).

  void InitLookAheadFst(const Fst<Arc> &fst, bool copy = false) override {}

  bool LookAheadFst(const Fst<Arc> &, StateId) final { return true; }

  bool LookAheadLabel(Label) const final { return true; }

  bool LookAheadPrefix(Arc *) const { return false; }

  Weight LookAheadWeight() const { return Weight::One(); }

 private:
  M matcher_;
};

// Look-ahead of one transition. Template argument flags accepts flags to
// control behavior.
template <class M,
          uint32_t flags = kLookAheadNonEpsilons | kLookAheadEpsilons |
                         kLookAheadWeight | kLookAheadPrefix>
class ArcLookAheadMatcher : public LookAheadMatcherBase<typename M::FST::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using MatcherData = NullAddOn;

  using LookAheadMatcherBase<Arc>::ClearLookAheadWeight;
  using LookAheadMatcherBase<Arc>::LookAheadWeight;
  using LookAheadMatcherBase<Arc>::SetLookAheadWeight;
  using LookAheadMatcherBase<Arc>::ClearLookAheadPrefix;
  using LookAheadMatcherBase<Arc>::LookAheadPrefix;
  using LookAheadMatcherBase<Arc>::SetLookAheadPrefix;

  enum : uint32_t { kFlags = flags };

  // This makes a copy of the FST.
  ArcLookAheadMatcher(
      const FST &fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::shared_ptr<MatcherData>())
      : matcher_(fst, match_type),
        fst_(matcher_.GetFst()),
        lfst_(nullptr),
        state_(kNoStateId) {}

  // This doesn't copy the FST.
  ArcLookAheadMatcher(
      const FST *fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::shared_ptr<MatcherData>())
      : matcher_(fst, match_type),
        fst_(matcher_.GetFst()),
        lfst_(nullptr),
        state_(kNoStateId) {}

  // This makes a copy of the FST.
  ArcLookAheadMatcher(const ArcLookAheadMatcher<M, flags> &lmatcher,
                      bool safe = false)
      : matcher_(lmatcher.matcher_, safe),
        fst_(matcher_.GetFst()),
        lfst_(lmatcher.lfst_),
        state_(kNoStateId) {}

  // General matcher methods.
  ArcLookAheadMatcher<M, flags> *Copy(bool safe = false) const override {
    return new ArcLookAheadMatcher<M, flags>(*this, safe);
  }

  MatchType Type(bool test) const override { return matcher_.Type(test); }

  void SetState(StateId s) final {
    state_ = s;
    matcher_.SetState(s);
  }

  bool Find(Label label) final { return matcher_.Find(label); }

  bool Done() const final { return matcher_.Done(); }

  const Arc &Value() const final { return matcher_.Value(); }

  void Next() final { matcher_.Next(); }

  Weight Final(StateId s) const final { return matcher_.Final(s); }

  std::ptrdiff_t Priority(StateId s) final { return matcher_.Priority(s); }

  const FST &GetFst() const override { return fst_; }

  uint64_t Properties(uint64_t props) const override {
    return matcher_.Properties(props);
  }

  uint32_t Flags() const override {
    return matcher_.Flags() | kInputLookAheadMatcher | kOutputLookAheadMatcher |
           kFlags;
  }

  const MatcherData *GetData() const { return nullptr; }

  std::shared_ptr<MatcherData> GetSharedData() const {
    return std::shared_ptr<MatcherData>();
  }

  // Look-ahead methods.

  void InitLookAheadFst(const Fst<Arc> &fst, bool copy = false) override {
    lfst_ = &fst;
  }

  // Checks if there is a matching (possibly super-final) transition
  // at (state_, s).
  bool LookAheadFst(const Fst<Arc> &, StateId) final;

  bool LookAheadLabel(Label label) const final { return matcher_.Find(label); }

 private:
  mutable M matcher_;
  const FST &fst_;        // Matcher FST.
  const Fst<Arc> *lfst_;  // Look-ahead FST.
  StateId state_;         // Matcher state.
};

template <class M, uint32_t flags>
bool ArcLookAheadMatcher<M, flags>::LookAheadFst(const Fst<Arc> &fst,
                                                 StateId s) {
  if (&fst != lfst_) InitLookAheadFst(fst);
  bool result = false;
  std::ptrdiff_t nprefix = 0;
  if (kFlags & kLookAheadWeight) ClearLookAheadWeight();
  if (kFlags & kLookAheadPrefix) ClearLookAheadPrefix();
  if (fst_.Final(state_) != Weight::Zero() &&
      lfst_->Final(s) != Weight::Zero()) {
    if (!(kFlags & (kLookAheadWeight | kLookAheadPrefix))) return true;
    ++nprefix;
    if (kFlags & kLookAheadWeight) {
      SetLookAheadWeight(
          Plus(LookAheadWeight(), Times(fst_.Final(state_), lfst_->Final(s))));
    }
    result = true;
  }
  if (matcher_.Find(kNoLabel)) {
    if (!(kFlags & (kLookAheadWeight | kLookAheadPrefix))) return true;
    ++nprefix;
    if (kFlags & kLookAheadWeight) {
      for (; !matcher_.Done(); matcher_.Next()) {
        SetLookAheadWeight(Plus(LookAheadWeight(), matcher_.Value().weight));
      }
    }
    result = true;
  }
  for (ArcIterator<Fst<Arc>> aiter(*lfst_, s); !aiter.Done(); aiter.Next()) {
    const auto &arc = aiter.Value();
    Label label = kNoLabel;
    switch (matcher_.Type(false)) {
      case MATCH_INPUT:
        label = arc.olabel;
        break;
      case MATCH_OUTPUT:
        label = arc.ilabel;
        break;
      default:
        FSTERROR() << "ArcLookAheadMatcher::LookAheadFst: Bad match type";
        return true;
    }
    if (label == 0) {
      if (!(kFlags & (kLookAheadWeight | kLookAheadPrefix))) return true;
      if (!(kFlags & kLookAheadNonEpsilonPrefix)) ++nprefix;
      if (kFlags & kLookAheadWeight) {
        SetLookAheadWeight(Plus(LookAheadWeight(), arc.weight));
      }
      result = true;
    } else if (matcher_.Find(label)) {
      if (!(kFlags & (kLookAheadWeight | kLookAheadPrefix))) return true;
      for (; !matcher_.Done(); matcher_.Next()) {
        ++nprefix;
        if (kFlags & kLookAheadWeight) {
          SetLookAheadWeight(Plus(LookAheadWeight(),
                                  Times(arc.weight, matcher_.Value().weight)));
        }
        if ((kFlags & kLookAheadPrefix) && nprefix == 1)
          SetLookAheadPrefix(arc);
      }
      result = true;
    }
  }
  if (kFlags & kLookAheadPrefix) {
    if (nprefix == 1) {
      ClearLookAheadWeight();  // Avoids double counting.
    } else {
      ClearLookAheadPrefix();
    }
  }
  return result;
}

// Template argument flags accepts flags to control behavior. It must include
// precisely one of kInputLookAheadMatcher or kOutputLookAheadMatcher.
template <class M,
          uint32_t flags = kLookAheadEpsilons | kLookAheadWeight |
                         kLookAheadPrefix | kLookAheadNonEpsilonPrefix |
                         kLookAheadKeepRelabelData,
          class Accumulator = DefaultAccumulator<typename M::Arc>,
          class Reachable = LabelReachable<typename M::Arc, Accumulator>>
class LabelLookAheadMatcher
    : public LookAheadMatcherBase<typename M::FST::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using MatcherData = typename Reachable::Data;

  using LookAheadMatcherBase<Arc>::ClearLookAheadWeight;
  using LookAheadMatcherBase<Arc>::LookAheadWeight;
  using LookAheadMatcherBase<Arc>::SetLookAheadWeight;
  using LookAheadMatcherBase<Arc>::ClearLookAheadPrefix;
  using LookAheadMatcherBase<Arc>::LookAheadPrefix;
  using LookAheadMatcherBase<Arc>::SetLookAheadPrefix;

  enum : uint32_t { kFlags = flags };

  // This makes a copy of the FST.
  LabelLookAheadMatcher(
      const FST &fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::shared_ptr<MatcherData>(),
      Accumulator *accumulator = nullptr)
      : matcher_(fst, match_type),
        lfst_(nullptr),
        state_(kNoStateId),
        error_(false) {
    Init(fst, match_type, data, accumulator);
  }

  // This doesn't copy the FST.
  LabelLookAheadMatcher(
      const FST *fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::shared_ptr<MatcherData>(),
      Accumulator *accumulator = nullptr)
      : matcher_(fst, match_type),
        lfst_(nullptr),
        state_(kNoStateId),
        error_(false) {
    Init(*fst, match_type, data, accumulator);
  }

  // This makes a copy of the FST.
  LabelLookAheadMatcher(
      const LabelLookAheadMatcher<M, flags, Accumulator, Reachable> &lmatcher,
      bool safe = false)
      : matcher_(lmatcher.matcher_, safe),
        lfst_(lmatcher.lfst_),
        label_reachable_(lmatcher.label_reachable_
                             ? new Reachable(*lmatcher.label_reachable_, safe)
                             : nullptr),
        state_(kNoStateId),
        error_(lmatcher.error_) {}

  LabelLookAheadMatcher<M, flags, Accumulator, Reachable> *Copy(
      bool safe = false) const override {
    return new LabelLookAheadMatcher<M, flags, Accumulator, Reachable>(*this,
                                                                       safe);
  }

  MatchType Type(bool test) const override { return matcher_.Type(test); }

  void SetState(StateId s) final {
    if (state_ == s) return;
    state_ = s;
    match_set_state_ = false;
    reach_set_state_ = false;
  }

  bool Find(Label label) final {
    if (!match_set_state_) {
      matcher_.SetState(state_);
      match_set_state_ = true;
    }
    return matcher_.Find(label);
  }

  bool Done() const final { return matcher_.Done(); }

  const Arc &Value() const final { return matcher_.Value(); }

  void Next() final { matcher_.Next(); }

  Weight Final(StateId s) const final { return matcher_.Final(s); }

  std::ptrdiff_t Priority(StateId s) final { return matcher_.Priority(s); }

  const FST &GetFst() const override { return matcher_.GetFst(); }

  uint64_t Properties(uint64_t inprops) const override {
    auto outprops = matcher_.Properties(inprops);
    if (error_ || (label_reachable_ && label_reachable_->Error())) {
      outprops |= kError;
    }
    return outprops;
  }

  uint32_t Flags() const override {
    if (label_reachable_ && label_reachable_->GetData()->ReachInput()) {
      return matcher_.Flags() | kFlags | kInputLookAheadMatcher;
    } else if (label_reachable_ && !label_reachable_->GetData()->ReachInput()) {
      return matcher_.Flags() | kFlags | kOutputLookAheadMatcher;
    } else {
      return matcher_.Flags();
    }
  }

  const MatcherData *GetData() const {
    return label_reachable_ ? label_reachable_->GetData() : nullptr;
  };

  std::shared_ptr<MatcherData> GetSharedData() const {
    return label_reachable_ ? label_reachable_->GetSharedData()
                            : std::shared_ptr<MatcherData>();
  }
  // Checks if there is a matching (possibly super-final) transition at
  // (state_, s).
  template <class LFST>
  bool LookAheadFst(const LFST &fst, StateId s);

  // Required to make class concrete.
  bool LookAheadFst(const Fst<Arc> &fst, StateId s) final {
    return LookAheadFst<Fst<Arc>>(fst, s);
  }

  void InitLookAheadFst(const Fst<Arc> &fst, bool copy = false) override {
    lfst_ = &fst;
    if (label_reachable_) {
      const bool reach_input = Type(false) == MATCH_OUTPUT;
      label_reachable_->ReachInit(fst, reach_input, copy);
    }
  }

  template <class LFST>
  void InitLookAheadFst(const LFST &fst, bool copy = false) {
    lfst_ = static_cast<const Fst<Arc> *>(&fst);
    if (label_reachable_) {
      const bool reach_input = Type(false) == MATCH_OUTPUT;
      label_reachable_->ReachInit(fst, reach_input, copy);
    }
  }

  bool LookAheadLabel(Label label) const final {
    if (label == 0) return true;
    if (label_reachable_) {
      if (!reach_set_state_) {
        label_reachable_->SetState(state_);
        reach_set_state_ = true;
      }
      return label_reachable_->Reach(label);
    } else {
      return true;
    }
  }

 private:
  void Init(const FST &fst, MatchType match_type,
            std::shared_ptr<MatcherData> data,
            Accumulator *accumulator) {
    if (!(kFlags & (kInputLookAheadMatcher | kOutputLookAheadMatcher))) {
      FSTERROR() << "LabelLookaheadMatcher: Bad matcher flags: " << kFlags;
      error_ = true;
    }
    const bool reach_input = match_type == MATCH_INPUT;
    if (data) {
      if (reach_input == data->ReachInput()) {
        label_reachable_.reset(new Reachable(data, accumulator));
      }
    } else if ((reach_input && (kFlags & kInputLookAheadMatcher)) ||
               (!reach_input && (kFlags & kOutputLookAheadMatcher))) {
      label_reachable_.reset(new Reachable(fst, reach_input, accumulator,
                                           kFlags & kLookAheadKeepRelabelData));
    }
  }

  mutable M matcher_;
  const Fst<Arc> *lfst_;                        // Look-ahead FST.
  std::unique_ptr<Reachable> label_reachable_;  // Label reachability info.
  StateId state_;                               // Matcher state.
  bool match_set_state_;                        // matcher_.SetState called?
  mutable bool reach_set_state_;                // reachable_.SetState called?
  bool error_;                                  // Error encountered?
};

template <class M, uint32_t flags, class Accumulator, class Reachable>
template <class LFST>
inline bool LabelLookAheadMatcher<M, flags, Accumulator,
                                  Reachable>::LookAheadFst(const LFST &fst,
                                                           StateId s) {
  if (static_cast<const Fst<Arc> *>(&fst) != lfst_) InitLookAheadFst(fst);
  ClearLookAheadWeight();
  ClearLookAheadPrefix();
  if (!label_reachable_) return true;
  label_reachable_->SetState(state_, s);
  reach_set_state_ = true;
  bool compute_weight = kFlags & kLookAheadWeight;
  bool compute_prefix = kFlags & kLookAheadPrefix;
  ArcIterator<LFST> aiter(fst, s);
  aiter.SetFlags(kArcNoCache, kArcNoCache);  // Makes caching optional.
  const bool reach_arc = label_reachable_->Reach(
      &aiter, 0, internal::NumArcs(*lfst_, s), compute_weight);
  const auto lfinal = internal::Final(*lfst_, s);
  const bool reach_final =
      lfinal != Weight::Zero() && label_reachable_->ReachFinal();
  if (reach_arc) {
    const auto begin = label_reachable_->ReachBegin();
    const auto end = label_reachable_->ReachEnd();
    if (compute_prefix && end - begin == 1 && !reach_final) {
      aiter.Seek(begin);
      SetLookAheadPrefix(aiter.Value());
      compute_weight = false;
    } else if (compute_weight) {
      SetLookAheadWeight(label_reachable_->ReachWeight());
    }
  }
  if (reach_final && compute_weight) {
    SetLookAheadWeight(reach_arc ? Plus(LookAheadWeight(), lfinal) : lfinal);
  }
  return reach_arc || reach_final;
}

// Label-lookahead relabeling class.
template <class Arc, class Data = LabelReachableData<typename Arc::Label>>
class LabelLookAheadRelabeler {
 public:
  using Label = typename Arc::Label;
  using Reachable = LabelReachable<Arc, DefaultAccumulator<Arc>, Data>;

  // Relabels matcher FST (initialization function object).
  template <typename Impl>
  explicit LabelLookAheadRelabeler(std::shared_ptr<Impl> *impl);

  // Relabels arbitrary FST. Class LFST should be a label-lookahead FST.
  template <class LFST>
  static void Relabel(MutableFst<Arc> *fst, const LFST &mfst,
                      bool relabel_input) {
    const auto *data = mfst.GetAddOn();
    Reachable reachable(data->First() ? data->SharedFirst()
                                      : data->SharedSecond());
    reachable.Relabel(fst, relabel_input);
  }

  // Returns relabeling pairs (cf. relabel.h::Relabel()). Class LFST should be a
  // label-lookahead FST. If avoid_collisions is true, extra pairs are added to
  // ensure no collisions when relabeling automata that have labels unseen here.
  template <class LFST>
  static void RelabelPairs(const LFST &mfst,
                           std::vector<std::pair<Label, Label>> *pairs,
                           bool avoid_collisions = false) {
    const auto *data = mfst.GetAddOn();
    Reachable reachable(data->First() ? data->SharedFirst()
                                      : data->SharedSecond());
    reachable.RelabelPairs(pairs, avoid_collisions);
  }
};

template <class Arc, class Data>
template <typename Impl>
inline LabelLookAheadRelabeler<Arc, Data>::LabelLookAheadRelabeler(
    std::shared_ptr<Impl> *impl) {
  Fst<Arc> &fst = (*impl)->GetFst();
  auto data = (*impl)->GetSharedAddOn();
  const auto name = (*impl)->Type();
  const bool is_mutable = fst.Properties(kMutable, false);
  std::unique_ptr<MutableFst<Arc>> mfst;
  if (is_mutable) {
    mfst.reset(static_cast<MutableFst<Arc> *>(&fst));
  } else {
    mfst.reset(new VectorFst<Arc>(fst));
  }
  if (data->First()) {  // reach_input.
    Reachable reachable(data->SharedFirst());
    reachable.Relabel(mfst.get(), true);
    if (!FLAGS_save_relabel_ipairs.empty()) {
      std::vector<std::pair<Label, Label>> pairs;
      reachable.RelabelPairs(&pairs, true);
      WriteLabelPairs(FLAGS_save_relabel_ipairs, pairs);
    }
  } else {
    Reachable reachable(data->SharedSecond());
    reachable.Relabel(mfst.get(), false);
    if (!FLAGS_save_relabel_opairs.empty()) {
      std::vector<std::pair<Label, Label>> pairs;
      reachable.RelabelPairs(&pairs, true);
      WriteLabelPairs(FLAGS_save_relabel_opairs, pairs);
    }
  }
  if (!is_mutable) {
    *impl = std::make_shared<Impl>(*mfst, name);
    (*impl)->SetAddOn(data);
  }
}

// Generic lookahead matcher, templated on the FST definition (a wrapper around
// a pointer to specific one).
template <class F>
class LookAheadMatcher {
 public:
  using FST = F;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using LBase = LookAheadMatcherBase<Arc>;

  // This makes a copy of the FST.
  LookAheadMatcher(const FST &fst, MatchType match_type)
      : owned_fst_(fst.Copy()),
        base_(owned_fst_->InitMatcher(match_type)),
        lookahead_(false) {
    if (!base_) base_.reset(new SortedMatcher<FST>(owned_fst_.get(),
                                                   match_type));
  }

  // This doesn't copy the FST.
  LookAheadMatcher(const FST *fst, MatchType match_type)
      : base_(fst->InitMatcher(match_type)),
        lookahead_(false) {
    if (!base_) base_.reset(new SortedMatcher<FST>(fst, match_type));
  }

  // This makes a copy of the FST.
  LookAheadMatcher(const LookAheadMatcher<FST> &matcher, bool safe = false)
      : base_(matcher.base_->Copy(safe)),
        lookahead_(matcher.lookahead_) { }

  // Takes ownership of base.
  explicit LookAheadMatcher(MatcherBase<Arc> *base)
      : base_(base), lookahead_(false) {}

  LookAheadMatcher<FST> *Copy(bool safe = false) const {
    return new LookAheadMatcher<FST>(*this, safe);
  }

  MatchType Type(bool test) const { return base_->Type(test); }

  void SetState(StateId s) { base_->SetState(s); }

  bool Find(Label label) { return base_->Find(label); }

  bool Done() const { return base_->Done(); }

  const Arc &Value() const { return base_->Value(); }

  void Next() { base_->Next(); }

  Weight Final(StateId s) const { return base_->Final(s); }

  std::ptrdiff_t Priority(StateId s) { return base_->Priority(s); }

  const FST &GetFst() const {
    return static_cast<const FST &>(base_->GetFst());
  }

  uint64_t Properties(uint64_t props) const { return base_->Properties(props); }

  uint32_t Flags() const { return base_->Flags(); }

  bool LookAheadLabel(Label label) const {
    if (LookAheadCheck()) {
      return static_cast<LBase *>(base_.get())->LookAheadLabel(label);
    } else {
      return true;
    }
  }

  bool LookAheadFst(const Fst<Arc> &fst, StateId s) {
    if (LookAheadCheck()) {
      return static_cast<LBase *>(base_.get())->LookAheadFst(fst, s);
    } else {
      return true;
    }
  }

  Weight LookAheadWeight() const {
    if (LookAheadCheck()) {
      return static_cast<LBase *>(base_.get())->LookAheadWeight();
    } else {
      return Weight::One();
    }
  }

  bool LookAheadPrefix(Arc *arc) const {
    if (LookAheadCheck()) {
      return static_cast<LBase *>(base_.get())->LookAheadPrefix(arc);
    } else {
      return false;
    }
  }

  void InitLookAheadFst(const Fst<Arc> &fst, bool copy = false) {
    if (LookAheadCheck()) {
      static_cast<LBase *>(base_.get())->InitLookAheadFst(fst, copy);
    }
  }

 private:
  bool LookAheadCheck() const {
    if (!lookahead_) {
      lookahead_ =
          base_->Flags() & (kInputLookAheadMatcher | kOutputLookAheadMatcher);
      if (!lookahead_) {
        FSTERROR() << "LookAheadMatcher: No look-ahead matcher defined";
      }
    }
    return lookahead_;
  }

  std::unique_ptr<const FST> owned_fst_;
  std::unique_ptr<MatcherBase<Arc>> base_;
  mutable bool lookahead_;

  LookAheadMatcher &operator=(const LookAheadMatcher &) = delete;
};

}  // namespace fst

#endif  // FST_LOOKAHEAD_MATCHER_H_
