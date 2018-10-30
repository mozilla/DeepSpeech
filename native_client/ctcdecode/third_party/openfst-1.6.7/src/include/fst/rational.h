// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// An FST implementation and base interface for delayed unions, concatenations,
// and closures.

#ifndef FST_RATIONAL_H_
#define FST_RATIONAL_H_

#include <algorithm>
#include <string>
#include <vector>

#include <fst/mutable-fst.h>
#include <fst/replace.h>
#include <fst/test-properties.h>


namespace fst {

using RationalFstOptions = CacheOptions;

// This specifies whether to add the empty string.
enum ClosureType {
  CLOSURE_STAR = 0,  // Add the empty string.
  CLOSURE_PLUS = 1   // Don't add the empty string.
};

template <class Arc>
class RationalFst;

template <class Arc>
void Union(RationalFst<Arc> *fst1, const Fst<Arc> &fst2);

template <class Arc>
void Concat(RationalFst<Arc> *fst1, const Fst<Arc> &fst2);

template <class Arc>
void Concat(const Fst<Arc> &fst1, RationalFst<Arc> *fst2);

template <class Arc>
void Closure(RationalFst<Arc> *fst, ClosureType closure_type);

namespace internal {

// Implementation class for delayed unions, concatenations and closures.
template <class A>
class RationalFstImpl : public FstImpl<A> {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::WriteHeader;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  explicit RationalFstImpl(const RationalFstOptions &opts)
      : nonterminals_(0), replace_options_(opts, 0) {
    SetType("rational");
    fst_tuples_.push_back(std::make_pair(0, nullptr));
  }

  RationalFstImpl(const RationalFstImpl<Arc> &impl)
      : rfst_(impl.rfst_),
        nonterminals_(impl.nonterminals_),
        replace_(impl.replace_ ? impl.replace_->Copy(true) : nullptr),
        replace_options_(impl.replace_options_) {
    SetType("rational");
    fst_tuples_.reserve(impl.fst_tuples_.size());
    for (const auto &pair : impl.fst_tuples_) {
      fst_tuples_.emplace_back(pair.first,
                               pair.second ? pair.second->Copy(true) : nullptr);
    }
  }

  ~RationalFstImpl() override {
    for (auto &tuple : fst_tuples_) delete tuple.second;
  }

  StateId Start() { return Replace()->Start(); }

  Weight Final(StateId s) { return Replace()->Final(s); }

  size_t NumArcs(StateId s) { return Replace()->NumArcs(s); }

  size_t NumInputEpsilons(StateId s) { return Replace()->NumInputEpsilons(s); }

  size_t NumOutputEpsilons(StateId s) {
    return Replace()->NumOutputEpsilons(s);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) && Replace()->Properties(kError, false)) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  // Implementation of UnionFst(fst1, fst2).
  void InitUnion(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {
    replace_.reset();
    const auto props1 = fst1.Properties(kFstProperties, false);
    const auto props2 = fst2.Properties(kFstProperties, false);
    SetInputSymbols(fst1.InputSymbols());
    SetOutputSymbols(fst1.OutputSymbols());
    rfst_.AddState();
    rfst_.AddState();
    rfst_.SetStart(0);
    rfst_.SetFinal(1, Weight::One());
    rfst_.SetInputSymbols(fst1.InputSymbols());
    rfst_.SetOutputSymbols(fst1.OutputSymbols());
    nonterminals_ = 2;
    rfst_.AddArc(0, Arc(0, -1, Weight::One(), 1));
    rfst_.AddArc(0, Arc(0, -2, Weight::One(), 1));
    fst_tuples_.push_back(std::make_pair(-1, fst1.Copy()));
    fst_tuples_.push_back(std::make_pair(-2, fst2.Copy()));
    SetProperties(UnionProperties(props1, props2, true), kCopyProperties);
  }

  // Implementation of ConcatFst(fst1, fst2).
  void InitConcat(const Fst<Arc> &fst1, const Fst<Arc> &fst2) {
    replace_.reset();
    const auto props1 = fst1.Properties(kFstProperties, false);
    const auto props2 = fst2.Properties(kFstProperties, false);
    SetInputSymbols(fst1.InputSymbols());
    SetOutputSymbols(fst1.OutputSymbols());
    rfst_.AddState();
    rfst_.AddState();
    rfst_.AddState();
    rfst_.SetStart(0);
    rfst_.SetFinal(2, Weight::One());
    rfst_.SetInputSymbols(fst1.InputSymbols());
    rfst_.SetOutputSymbols(fst1.OutputSymbols());
    nonterminals_ = 2;
    rfst_.AddArc(0, Arc(0, -1, Weight::One(), 1));
    rfst_.AddArc(1, Arc(0, -2, Weight::One(), 2));
    fst_tuples_.push_back(std::make_pair(-1, fst1.Copy()));
    fst_tuples_.push_back(std::make_pair(-2, fst2.Copy()));
    SetProperties(ConcatProperties(props1, props2, true), kCopyProperties);
  }

  // Implementation of ClosureFst(fst, closure_type).
  void InitClosure(const Fst<Arc> &fst, ClosureType closure_type) {
    replace_.reset();
    const auto props = fst.Properties(kFstProperties, false);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
    if (closure_type == CLOSURE_STAR) {
      rfst_.AddState();
      rfst_.SetStart(0);
      rfst_.SetFinal(0, Weight::One());
      rfst_.AddArc(0, Arc(0, -1, Weight::One(), 0));
    } else {
      rfst_.AddState();
      rfst_.AddState();
      rfst_.SetStart(0);
      rfst_.SetFinal(1, Weight::One());
      rfst_.AddArc(0, Arc(0, -1, Weight::One(), 1));
      rfst_.AddArc(1, Arc(0, 0, Weight::One(), 0));
    }
    rfst_.SetInputSymbols(fst.InputSymbols());
    rfst_.SetOutputSymbols(fst.OutputSymbols());
    fst_tuples_.push_back(std::make_pair(-1, fst.Copy()));
    nonterminals_ = 1;
    SetProperties(ClosureProperties(props, closure_type == CLOSURE_STAR, true),
                  kCopyProperties);
  }

  // Implementation of Union(Fst &, RationalFst *).
  void AddUnion(const Fst<Arc> &fst) {
    replace_.reset();
    const auto props1 = FstImpl<A>::Properties();
    const auto props2 = fst.Properties(kFstProperties, false);
    VectorFst<Arc> afst;
    afst.AddState();
    afst.AddState();
    afst.SetStart(0);
    afst.SetFinal(1, Weight::One());
    ++nonterminals_;
    afst.AddArc(0, Arc(0, -nonterminals_, Weight::One(), 1));
    Union(&rfst_, afst);
    fst_tuples_.push_back(std::make_pair(-nonterminals_, fst.Copy()));
    SetProperties(UnionProperties(props1, props2, true), kCopyProperties);
  }

  // Implementation of Concat(Fst &, RationalFst *).
  void AddConcat(const Fst<Arc> &fst, bool append) {
    replace_.reset();
    const auto props1 = FstImpl<A>::Properties();
    const auto props2 = fst.Properties(kFstProperties, false);
    VectorFst<Arc> afst;
    afst.AddState();
    afst.AddState();
    afst.SetStart(0);
    afst.SetFinal(1, Weight::One());
    ++nonterminals_;
    afst.AddArc(0, Arc(0, -nonterminals_, Weight::One(), 1));
    if (append) {
      Concat(&rfst_, afst);
    } else {
      Concat(afst, &rfst_);
    }
    fst_tuples_.push_back(std::make_pair(-nonterminals_, fst.Copy()));
    SetProperties(ConcatProperties(props1, props2, true), kCopyProperties);
  }

  // Implementation of Closure(RationalFst *, closure_type).
  void AddClosure(ClosureType closure_type) {
    replace_.reset();
    const auto props = FstImpl<A>::Properties();
    Closure(&rfst_, closure_type);
    SetProperties(ClosureProperties(props, closure_type == CLOSURE_STAR, true),
                  kCopyProperties);
  }

  // Returns the underlying ReplaceFst, preserving ownership of the underlying
  // object.
  ReplaceFst<Arc> *Replace() const {
    if (!replace_) {
      fst_tuples_[0].second = rfst_.Copy();
      replace_.reset(new ReplaceFst<Arc>(fst_tuples_, replace_options_));
    }
    return replace_.get();
  }

 private:
  // Rational topology machine, using negative non-terminals.
  VectorFst<Arc> rfst_;
  // Number of nonterminals used.
  Label nonterminals_;
  // Contains the nonterminals and their corresponding FSTs.
  mutable std::vector<std::pair<Label, const Fst<Arc> *>> fst_tuples_;
  // Underlying ReplaceFst.
  mutable std::unique_ptr<ReplaceFst<Arc>> replace_;
  const ReplaceFstOptions<Arc> replace_options_;
};

}  // namespace internal

// Parent class for the delayed rational operations (union, concatenation, and
// closure). This class attaches interface to implementation and handles
// reference counting, delegating most methods to ImplToFst.
template <class A>
class RationalFst : public ImplToFst<internal::RationalFstImpl<A>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using Impl = internal::RationalFstImpl<Arc>;

  friend class StateIterator<RationalFst<Arc>>;
  friend class ArcIterator<RationalFst<Arc>>;
  friend void Union<>(RationalFst<Arc> *fst1, const Fst<Arc> &fst2);
  friend void Concat<>(RationalFst<Arc> *fst1, const Fst<Arc> &fst2);
  friend void Concat<>(const Fst<Arc> &fst1, RationalFst<Arc> *fst2);
  friend void Closure<>(RationalFst<Arc> *fst, ClosureType closure_type);

  void InitStateIterator(StateIteratorData<Arc> *data) const override {
    GetImpl()->Replace()->InitStateIterator(data);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetImpl()->Replace()->InitArcIterator(s, data);
  }

 protected:
  using ImplToFst<Impl>::GetImpl;

  explicit RationalFst(const RationalFstOptions &opts = RationalFstOptions())
      : ImplToFst<Impl>(std::make_shared<Impl>(opts)) {}

  // See Fst<>::Copy() for doc.
  RationalFst(const RationalFst<Arc> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

 private:
  RationalFst &operator=(const RationalFst &) = delete;
};

// Specialization for RationalFst.
template <class Arc>
class StateIterator<RationalFst<Arc>> : public StateIterator<ReplaceFst<Arc>> {
 public:
  explicit StateIterator(const RationalFst<Arc> &fst)
      : StateIterator<ReplaceFst<Arc>>(*(fst.GetImpl()->Replace())) {}
};

// Specialization for RationalFst.
template <class Arc>
class ArcIterator<RationalFst<Arc>> : public CacheArcIterator<ReplaceFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const RationalFst<Arc> &fst, StateId s)
      : ArcIterator<ReplaceFst<Arc>>(*(fst.GetImpl()->Replace()), s) {}
};

}  // namespace fst

#endif  // FST_RATIONAL_H_
