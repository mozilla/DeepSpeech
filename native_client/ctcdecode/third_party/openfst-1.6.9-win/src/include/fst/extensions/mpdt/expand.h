// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Expands an MPDT to an FST.

#ifndef FST_EXTENSIONS_MPDT_EXPAND_H_
#define FST_EXTENSIONS_MPDT_EXPAND_H_

#include <vector>

#include <fst/extensions/mpdt/mpdt.h>
#include <fst/extensions/pdt/paren.h>
#include <fst/cache.h>
#include <fst/mutable-fst.h>
#include <fst/queue.h>
#include <fst/state-table.h>
#include <fst/test-properties.h>

namespace fst {

template <class Arc>
struct MPdtExpandFstOptions : public CacheOptions {
  bool keep_parentheses;
  internal::MPdtStack<typename Arc::StateId, typename Arc::Label> *stack;
  PdtStateTable<typename Arc::StateId, typename Arc::StateId> *state_table;

  MPdtExpandFstOptions(
      const CacheOptions &opts = CacheOptions(), bool kp = false,
      internal::MPdtStack<typename Arc::StateId, typename Arc::Label> *s =
          nullptr,
      PdtStateTable<typename Arc::StateId, typename Arc::StateId> *st = nullptr)
      : CacheOptions(opts), keep_parentheses(kp), stack(s), state_table(st) {}
};

// Properties for an expanded PDT.
inline uint64_t MPdtExpandProperties(uint64_t inprops) {
  return inprops & (kAcceptor | kAcyclic | kInitialAcyclic | kUnweighted);
}

namespace internal {

// Implementation class for ExpandFst
template <class Arc>
class MPdtExpandFstImpl : public CacheImpl<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StackId = StateId;
  using StateTuple = PdtStateTuple<StateId, StackId>;
  using ParenStack = internal::MPdtStack<StateId, Label>;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  using CacheBaseImpl<CacheState<Arc>>::PushArc;
  using CacheBaseImpl<CacheState<Arc>>::HasArcs;
  using CacheBaseImpl<CacheState<Arc>>::HasFinal;
  using CacheBaseImpl<CacheState<Arc>>::HasStart;
  using CacheBaseImpl<CacheState<Arc>>::SetArcs;
  using CacheBaseImpl<CacheState<Arc>>::SetFinal;
  using CacheBaseImpl<CacheState<Arc>>::SetStart;

  MPdtExpandFstImpl(const Fst<Arc> &fst,
                    const std::vector<std::pair<Label, Label>> &parens,
                    const std::vector<Label> &assignments,
                    const MPdtExpandFstOptions<Arc> &opts)
      : CacheImpl<Arc>(opts),
        fst_(fst.Copy()),
        stack_(opts.stack ? opts.stack : new ParenStack(parens, assignments)),
        state_table_(opts.state_table ? opts.state_table
                                      : new PdtStateTable<StateId, StackId>()),
        own_stack_(!opts.stack),
        own_state_table_(!opts.state_table),
        keep_parentheses_(opts.keep_parentheses) {
    SetType("expand");
    const auto props = fst.Properties(kFstProperties, false);
    SetProperties(MPdtExpandProperties(props), kCopyProperties);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  MPdtExpandFstImpl(const MPdtExpandFstImpl &impl)
      : CacheImpl<Arc>(impl),
        fst_(impl.fst_->Copy(true)),
        stack_(new ParenStack(*impl.stack_)),
        state_table_(new PdtStateTable<StateId, StackId>()),
        own_stack_(true),
        own_state_table_(true),
        keep_parentheses_(impl.keep_parentheses_) {
    SetType("expand");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  ~MPdtExpandFstImpl() override {
    if (own_stack_) delete stack_;
    if (own_state_table_) delete state_table_;
  }

  StateId Start() {
    if (!HasStart()) {
      const auto s = fst_->Start();
      if (s == kNoStateId) return kNoStateId;
      const StateTuple tuple(s, 0);
      const auto start = state_table_->FindState(tuple);
      SetStart(start);
    }
    return CacheImpl<Arc>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      const auto &tuple = state_table_->Tuple(s);
      const auto weight = fst_->Final(tuple.state_id);
      SetFinal(s,
               (weight != Weight::Zero() && tuple.stack_id == 0)
                   ? weight
                   : Weight::Zero());
    }
    return CacheImpl<Arc>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) ExpandState(s);
    return CacheImpl<Arc>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) ExpandState(s);
    return CacheImpl<Arc>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) ExpandState(s);
    return CacheImpl<Arc>::NumOutputEpsilons(s);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) ExpandState(s);
    CacheImpl<Arc>::InitArcIterator(s, data);
  }

  // Computes the outgoing transitions from a state, creating new destination
  // states as needed.
  void ExpandState(StateId s) {
    const auto tuple = state_table_->Tuple(s);
    for (ArcIterator<Fst<Arc>> aiter(*fst_, tuple.state_id); !aiter.Done();
         aiter.Next()) {
      auto arc = aiter.Value();
      const auto stack_id = stack_->Find(tuple.stack_id, arc.ilabel);
      if (stack_id == -1) {
        continue;  // Non-matching close parenthesis.
      } else if ((stack_id != tuple.stack_id) && !keep_parentheses_) {
        arc.ilabel = arc.olabel = 0;  // Stack push/pop.
      }
      const StateTuple ntuple(arc.nextstate, stack_id);
      arc.nextstate = state_table_->FindState(ntuple);
      PushArc(s, arc);
    }
    SetArcs(s);
  }

  const ParenStack &GetStack() const { return *stack_; }

  const PdtStateTable<StateId, StackId> &GetStateTable() const {
    return *state_table_;
  }

 private:
  std::unique_ptr<const Fst<Arc>> fst_;
  ParenStack *stack_;
  PdtStateTable<StateId, StackId> *state_table_;
  const bool own_stack_;
  const bool own_state_table_;
  const bool keep_parentheses_;

  MPdtExpandFstImpl &operator=(const MPdtExpandFstImpl &) = delete;
};

}  // namespace internal

// Expands a multi-pushdown transducer (MPDT) encoded as an FST into an FST.
// This version is a delayed FST. In the MPDT, some transitions are labeled with
// open or close parentheses. To be interpreted as an MPDT, the parens for each
// stack must balance on a path. The open-close parenthesis label
// pairs are passed using the parens argument, and the assignment of those pairs
// to stacks is passed using the assignments argument. Expansion enforces the
// parenthesis constraints. The MPDT must be
// expandable as an FST.
//
// This class attaches interface to implementation and handles
// reference counting, delegating most methods to ImplToFst.
template <class A>
class MPdtExpandFst : public ImplToFst<internal::MPdtExpandFstImpl<A>> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using StackId = StateId;
  using ParenStack = internal::MPdtStack<StackId, Label>;
  using Store = DefaultCacheStore<Arc>;
  using State = typename Store::State;
  using Impl = internal::MPdtExpandFstImpl<Arc>;

  friend class ArcIterator<MPdtExpandFst<Arc>>;
  friend class StateIterator<MPdtExpandFst<Arc>>;

  MPdtExpandFst(const Fst<Arc> &fst,
                const std::vector<std::pair<Label, Label>> &parens,
                const std::vector<Label> &assignments)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, parens, assignments,
                                               MPdtExpandFstOptions<Arc>())) {}

  MPdtExpandFst(const Fst<Arc> &fst,
                const std::vector<std::pair<Label, Label>> &parens,
                const std::vector<Label> &assignments,
                const MPdtExpandFstOptions<Arc> &opts)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, parens, assignments, opts)) {}

  // See Fst<>::Copy() for doc.
  MPdtExpandFst(const MPdtExpandFst<Arc> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this ExpandFst. See Fst<>::Copy() for further doc.
  MPdtExpandFst<Arc> *Copy(bool safe = false) const override {
    return new MPdtExpandFst<A>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  const ParenStack &GetStack() const { return GetImpl()->GetStack(); }

  const PdtStateTable<StateId, StackId> &GetStateTable() const {
    return GetImpl()->GetStateTable();
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  void operator=(const MPdtExpandFst &) = delete;
};

// Specialization for MPdtExpandFst.
template <class Arc>
class StateIterator<MPdtExpandFst<Arc>>
    : public CacheStateIterator<MPdtExpandFst<Arc>> {
 public:
  explicit StateIterator(const MPdtExpandFst<Arc> &fst)
      : CacheStateIterator<MPdtExpandFst<Arc>>(fst, fst.GetMutableImpl()) {}
};

// Specialization for MPdtExpandFst.
template <class Arc>
class ArcIterator<MPdtExpandFst<Arc>>
    : public CacheArcIterator<MPdtExpandFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const MPdtExpandFst<Arc> &fst, StateId s)
      : CacheArcIterator<MPdtExpandFst<Arc>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->ExpandState(s);
  }
};

template <class Arc>
inline void MPdtExpandFst<Arc>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<MPdtExpandFst<Arc>>(*this);
}

struct MPdtExpandOptions {
  bool connect;
  bool keep_parentheses;

  explicit MPdtExpandOptions(bool connect = true, bool keep_parentheses = false)
      : connect(connect), keep_parentheses(keep_parentheses) {}
};

// Expands a multi-pushdown transducer (MPDT) encoded as an FST into an FST.
// This version writes the expanded PDT to a mutable FST. In the MPDT, some
// transitions are labeled with open or close parentheses. To be interpreted as
// an MPDT, the parens for each stack must balance on a path. The open-close
// parenthesis label pair sets are passed using the parens argument, and the
// assignment of those pairs to stacks is passed using the assignments argument.
// The expansion enforces the parenthesis constraints. The MPDT must be
// expandable as an FST.
template <class Arc>
void Expand(const Fst<Arc> &ifst,
            const std::vector<
            std::pair<typename Arc::Label, typename Arc::Label>> &parens,
            const std::vector<typename Arc::Label> &assignments,
            MutableFst<Arc> *ofst, const MPdtExpandOptions &opts) {
  MPdtExpandFstOptions<Arc> eopts;
  eopts.gc_limit = 0;
  eopts.keep_parentheses = opts.keep_parentheses;
  *ofst = MPdtExpandFst<Arc>(ifst, parens, assignments, eopts);
  if (opts.connect) Connect(ofst);
}

// Expands a multi-pushdown transducer (MPDT) encoded as an FST into an FST.
// This version writes the expanded PDT to a mutable FST. In the MPDT, some
// transitions are labeled with open or close parentheses. To be interpreted as
// an MPDT, the parens for each stack must balance on a path. The open-close
// parenthesis label pair sets are passed using the parens argument, and the
// assignment of those pairs to stacks is passed using the assignments argument.
// The expansion enforces the parenthesis constraints. The MPDT must be
// expandable as an FST.
template <class Arc>
void Expand(const Fst<Arc> &ifst,
            const std::vector<std::pair<typename Arc::Label,
            typename Arc::Label>> &parens,
            const std::vector<typename Arc::Label> &assignments,
            MutableFst<Arc> *ofst, bool connect = true,
            bool keep_parentheses = false) {
  const MPdtExpandOptions opts(connect, keep_parentheses);
  Expand(ifst, parens, assignments, ofst, opts);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_MPDT_EXPAND_H_
