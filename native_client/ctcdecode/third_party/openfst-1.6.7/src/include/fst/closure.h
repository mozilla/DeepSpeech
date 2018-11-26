// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to compute the concatenative closure of an FST.

#ifndef FST_CLOSURE_H_
#define FST_CLOSURE_H_

#include <algorithm>
#include <vector>

#include <fst/mutable-fst.h>
#include <fst/rational.h>


namespace fst {

// Computes the concatenative closure. This version modifies its
// MutableFst input. If an FST transduces string x to y with weight a,
// then its closure transduces x to y with weight a, xx to yy with
// weight Times(a, a), xxx to yyy with with Times(Times(a, a), a),
// etc. If closure_type == CLOSURE_STAR, then the empty string is
// transduced to itself with weight Weight::One() as well.
//
// Complexity:
//
//   Time: O(V)
//   Space: O(V)
//
// where V is the number of states.
template <class Arc>
void Closure(MutableFst<Arc> *fst, ClosureType closure_type) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  const auto props = fst->Properties(kFstProperties, false);
  const auto start = fst->Start();
  for (StateIterator<MutableFst<Arc>> siter(*fst); !siter.Done();
       siter.Next()) {
    const auto s = siter.Value();
    const auto weight = fst->Final(s);
    if (weight != Weight::Zero()) fst->AddArc(s, Arc(0, 0, weight, start));
  }
  if (closure_type == CLOSURE_STAR) {
    fst->ReserveStates(fst->NumStates() + 1);
    const auto nstart = fst->AddState();
    fst->SetStart(nstart);
    fst->SetFinal(nstart, Weight::One());
    if (start != kNoLabel) fst->AddArc(nstart, Arc(0, 0, Weight::One(), start));
  }
  fst->SetProperties(ClosureProperties(props, closure_type == CLOSURE_STAR),
                     kFstProperties);
}

// Computes the concatenative closure. This version modifies its
// RationalFst input.
template <class Arc>
void Closure(RationalFst<Arc> *fst, ClosureType closure_type) {
  fst->GetMutableImpl()->AddClosure(closure_type);
}

struct ClosureFstOptions : RationalFstOptions {
  ClosureType type;

  ClosureFstOptions(const RationalFstOptions &opts,
                    ClosureType type = CLOSURE_STAR)
      : RationalFstOptions(opts), type(type) {}

  explicit ClosureFstOptions(ClosureType type = CLOSURE_STAR) : type(type) {}
};

// Computes the concatenative closure. This version is a delayed FST. If an FST
// transduces string x to y with weight a, then its closure transduces x to y
// with weight a, xx to yy with weight Times(a, a), xxx to yyy with weight
// Times(Times(a, a), a), etc. If closure_type == CLOSURE_STAR, then the empty
// string is transduced to itself with weight Weight::One() as well.
//
// Complexity:
//
//   Time: O(v)
//   Space: O(v)
//
// where v is the number of states visited. Constant time and space to visit an
// input state or arc is assumed and exclusive of caching.
template <class A>
class ClosureFst : public RationalFst<A> {
 public:
  using Arc = A;

  ClosureFst(const Fst<Arc> &fst, ClosureType closure_type) {
    GetMutableImpl()->InitClosure(fst, closure_type);
  }

  ClosureFst(const Fst<Arc> &fst, const ClosureFstOptions &opts)
      : RationalFst<A>(opts) {
    GetMutableImpl()->InitClosure(fst, opts.type);
  }

  // See Fst<>::Copy() for doc.
  ClosureFst(const ClosureFst<Arc> &fst, bool safe = false)
      : RationalFst<A>(fst, safe) {}

  // Gets a copy of this ClosureFst. See Fst<>::Copy() for further doc.
  ClosureFst<A> *Copy(bool safe = false) const override {
    return new ClosureFst<A>(*this, safe);
  }

 private:
  using ImplToFst<internal::RationalFstImpl<Arc>>::GetImpl;
  using ImplToFst<internal::RationalFstImpl<Arc>>::GetMutableImpl;
};

// Specialization for ClosureFst.
template <class Arc>
class StateIterator<ClosureFst<Arc>> : public StateIterator<RationalFst<Arc>> {
 public:
  explicit StateIterator(const ClosureFst<Arc> &fst)
      : StateIterator<RationalFst<Arc>>(fst) {}
};

// Specialization for ClosureFst.
template <class Arc>
class ArcIterator<ClosureFst<Arc>> : public ArcIterator<RationalFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const ClosureFst<Arc> &fst, StateId s)
      : ArcIterator<RationalFst<Arc>>(fst, s) {}
};

// Useful alias when using StdArc.
using StdClosureFst = ClosureFst<StdArc>;

}  // namespace fst

#endif  // FST_CLOSURE_H_
