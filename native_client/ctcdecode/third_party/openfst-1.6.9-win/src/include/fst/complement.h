// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to complement an FST.

#ifndef FST_COMPLEMENT_H_
#define FST_COMPLEMENT_H_

#include <algorithm>
#include <string>
#include <vector>
#include <fst/log.h>

#include <fst/fst.h>
#include <fst/test-properties.h>


namespace fst {

template <class Arc>
class ComplementFst;

namespace internal {

// Implementation of delayed ComplementFst. The algorithm used completes the
// (deterministic) FSA and then exchanges final and non-final states.
// Completion, i.e. ensuring that all labels can be read from every state, is
// accomplished by using ρ-labels, which match all labels that are otherwise
// not found leaving a state. The first state in the output is reserved to be a
// new state that is the destination of all ρ-labels. Each remaining output
// state s corresponds to input state s - 1. The first arc in the output at
// these states is the ρ-label, the remaining arcs correspond to the input
// arcs.
template <class A>
class ComplementFstImpl : public FstImpl<A> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<A>::SetType;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;

  friend class StateIterator<ComplementFst<Arc>>;
  friend class ArcIterator<ComplementFst<Arc>>;

  explicit ComplementFstImpl(const Fst<Arc> &fst) : fst_(fst.Copy()) {
    SetType("complement");
    uint64_t props = fst.Properties(kILabelSorted, false);
    SetProperties(ComplementProperties(props), kCopyProperties);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  ComplementFstImpl(const ComplementFstImpl<Arc> &impl)
      : fst_(impl.fst_->Copy()) {
    SetType("complement");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  StateId Start() const {
    if (Properties(kError)) return kNoStateId;
    auto start = fst_->Start();
    return start != kNoStateId ? start + 1 : 0;
  }

  // Exchange final and non-final states; makes ρ-destination state final.
  Weight Final(StateId s) const {
    if (s == 0 || fst_->Final(s - 1) == Weight::Zero()) {
      return Weight::One();
    } else {
      return Weight::Zero();
    }
  }

  size_t NumArcs(StateId s) const {
    return s == 0 ? 1 : fst_->NumArcs(s - 1) + 1;
  }

  size_t NumInputEpsilons(StateId s) const {
    return s == 0 ? 0 : fst_->NumInputEpsilons(s - 1);
  }

  size_t NumOutputEpsilons(StateId s) const {
    return s == 0 ? 0 : fst_->NumOutputEpsilons(s - 1);
  }

  uint64_t Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64_t Properties(uint64_t mask) const override {
    if ((mask & kError) && fst_->Properties(kError, false)) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

 private:
  std::unique_ptr<const Fst<Arc>> fst_;
};

}  // namespace internal

// Complements an automaton. This is a library-internal operation that
// introduces a (negative) ρ-label; use Difference/DifferenceFst in user code,
// which will not see this label. This version is a delayed FST.
//
// This class attaches interface to implementation and handles
// reference counting, delegating most methods to ImplToFst.
template <class A>
class ComplementFst : public ImplToFst<internal::ComplementFstImpl<A>> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Impl = internal::ComplementFstImpl<Arc>;

  friend class StateIterator<ComplementFst<Arc>>;
  friend class ArcIterator<ComplementFst<Arc>>;

  explicit ComplementFst(const Fst<Arc> &fst)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst)) {
    static constexpr auto props =
        kUnweighted | kNoEpsilons | kIDeterministic | kAcceptor;
    if (fst.Properties(props, true) != props) {
      FSTERROR() << "ComplementFst: Argument not an unweighted "
                 << "epsilon-free deterministic acceptor";
      GetImpl()->SetProperties(kError, kError);
    }
  }

  // See Fst<>::Copy() for doc.
  ComplementFst(const ComplementFst<Arc> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Gets a copy of this FST. See Fst<>::Copy() for further doc.
  ComplementFst<Arc> *Copy(bool safe = false) const override {
    return new ComplementFst<Arc>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  inline void InitArcIterator(StateId s,
                              ArcIteratorData<Arc> *data) const override;

  // Label that represents the ρ-transition; we use a negative value private to
  // the library and which will preserve FST label sort order.
  static const Label kRhoLabel = -2;

 private:
  using ImplToFst<Impl>::GetImpl;

  ComplementFst &operator=(const ComplementFst &) = delete;
};

template <class Arc>
const typename Arc::Label ComplementFst<Arc>::kRhoLabel;

// Specialization for ComplementFst.
template <class Arc>
class StateIterator<ComplementFst<Arc>> : public StateIteratorBase<Arc> {
 public:
  using StateId = typename Arc::StateId;

  explicit StateIterator(const ComplementFst<Arc> &fst)
      : siter_(*fst.GetImpl()->fst_), s_(0) {}

  bool Done() const final { return s_ > 0 && siter_.Done(); }

  StateId Value() const final { return s_; }

  void Next() final {
    if (s_ != 0) siter_.Next();
    ++s_;
  }

  void Reset() final {
    siter_.Reset();
    s_ = 0;
  }

 private:
  StateIterator<Fst<Arc>> siter_;
  StateId s_;
};

// Specialization for ComplementFst.
template <class Arc>
class ArcIterator<ComplementFst<Arc>> : public ArcIteratorBase<Arc> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  ArcIterator(const ComplementFst<Arc> &fst, StateId s) : s_(s), pos_(0) {
    if (s_ != 0) {
      aiter_.reset(new ArcIterator<Fst<Arc>>(*fst.GetImpl()->fst_, s - 1));
    }
  }

  bool Done() const final {
    if (s_ != 0) {
      return pos_ > 0 && aiter_->Done();
    } else {
      return pos_ > 0;
    }
  }

  // Adds the ρ-label to the ρ destination state.
  const Arc &Value() const final {
    if (pos_ == 0) {
      arc_.ilabel = arc_.olabel = ComplementFst<Arc>::kRhoLabel;
      arc_.weight = Weight::One();
      arc_.nextstate = 0;
    } else {
      arc_ = aiter_->Value();
      ++arc_.nextstate;
    }
    return arc_;
  }

  void Next() final {
    if (s_ != 0 && pos_ > 0) aiter_->Next();
    ++pos_;
  }

  size_t Position() const final { return pos_; }

  void Reset() final {
    if (s_ != 0) aiter_->Reset();
    pos_ = 0;
  }

  void Seek(size_t a) final {
    if (s_ != 0) {
      if (a == 0) {
        aiter_->Reset();
      } else {
        aiter_->Seek(a - 1);
      }
    }
    pos_ = a;
  }

  uint32_t Flags() const final { return kArcValueFlags; }

  void SetFlags(uint32_t, uint32_t) final {}

 private:
  std::unique_ptr<ArcIterator<Fst<Arc>>> aiter_;
  StateId s_;
  size_t pos_;
  mutable Arc arc_;
};

template <class Arc>
inline void ComplementFst<Arc>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<ComplementFst<Arc>>(*this);
}

template <class Arc>
inline void ComplementFst<Arc>::InitArcIterator(StateId s,
    ArcIteratorData<Arc> *data) const {
  data->base = new ArcIterator<ComplementFst<Arc>>(*this, s);
}

// Useful alias when using StdArc.
using StdComplementFst = ComplementFst<StdArc>;

}  // namespace fst

#endif  // FST_COMPLEMENT_H_
