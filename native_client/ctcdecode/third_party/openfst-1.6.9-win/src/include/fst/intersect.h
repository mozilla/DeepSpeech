// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to compute the intersection of two FSAs.

#ifndef FST_INTERSECT_H_
#define FST_INTERSECT_H_

#include <algorithm>
#include <vector>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/compose.h>


namespace fst {

using IntersectOptions = ComposeOptions;

template <class Arc, class M = Matcher<Fst<Arc>>,
          class Filter = SequenceComposeFilter<M>,
          class StateTable =
              GenericComposeStateTable<Arc, typename Filter::FilterState>>
struct IntersectFstOptions
    : public ComposeFstOptions<Arc, M, Filter, StateTable> {
  IntersectFstOptions() {}

  explicit IntersectFstOptions(const CacheOptions &opts, M *matcher1 = nullptr,
                               M *matcher2 = nullptr, Filter *filter = nullptr,
                               StateTable *state_table = nullptr)
      : ComposeFstOptions<Arc, M, Filter, StateTable>(opts, matcher1, matcher2,
                                                      filter, state_table) {}
};

// Computes the intersection (Hadamard product) of two FSAs. This version is a
// delayed FST. Only strings that are in both automata are retained in the
// result.
//
// The two arguments must be acceptors. One of the arguments must be
// label-sorted.
//
// Complexity: same as ComposeFst.
//
// Caveats: same as ComposeFst.
template <class A>
class IntersectFst : public ComposeFst<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using ComposeFst<A>::CreateBase;
  using ComposeFst<A>::CreateBase1;
  using ComposeFst<A>::Properties;

  IntersectFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
               const CacheOptions &opts = CacheOptions())
      : ComposeFst<Arc>(CreateBase(fst1, fst2, opts)) {
    const bool acceptors =
        fst1.Properties(kAcceptor, true) && fst2.Properties(kAcceptor, true);
    if (!acceptors) {
      FSTERROR() << "IntersectFst: Input FSTs are not acceptors";
      GetMutableImpl()->SetProperties(kError);
    }
  }

  template <class M, class Filter, class StateTable>
  IntersectFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
               const IntersectFstOptions<Arc, M, Filter, StateTable> &opts)
      : ComposeFst<Arc>(CreateBase1(fst1, fst2, opts)) {
    const bool acceptors =
        fst1.Properties(kAcceptor, true) && fst2.Properties(kAcceptor, true);
    if (!acceptors) {
      FSTERROR() << "IntersectFst: input FSTs are not acceptors";
      GetMutableImpl()->SetProperties(kError);
    }
  }

  // See Fst<>::Copy() for doc.
  IntersectFst(const IntersectFst<Arc> &fst, bool safe = false)
      : ComposeFst<Arc>(fst, safe) {}

  // Get a copy of this IntersectFst. See Fst<>::Copy() for further doc.
  IntersectFst<Arc> *Copy(bool safe = false) const override {
    return new IntersectFst<Arc>(*this, safe);
  }

 private:
  using ImplToFst<internal::ComposeFstImplBase<A>>::GetImpl;
  using ImplToFst<internal::ComposeFstImplBase<A>>::GetMutableImpl;
};

// Specialization for IntersectFst.
template <class Arc>
class StateIterator<IntersectFst<Arc>> : public StateIterator<ComposeFst<Arc>> {
 public:
  explicit StateIterator(const IntersectFst<Arc> &fst)
      : StateIterator<ComposeFst<Arc>>(fst) {}
};

// Specialization for IntersectFst.
template <class Arc>
class ArcIterator<IntersectFst<Arc>> : public ArcIterator<ComposeFst<Arc>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const IntersectFst<Arc> &fst, StateId s)
      : ArcIterator<ComposeFst<Arc>>(fst, s) {}
};

// Useful alias when using StdArc.
using StdIntersectFst = IntersectFst<StdArc>;

// Computes the intersection (Hadamard product) of two FSAs. This version
// writes the intersection to an output MurableFst. Only strings that are in
// both automata are retained in the result.
//
// The two arguments must be acceptors. One of the arguments must be
// label-sorted.
//
// Complexity: same as Compose.
//
// Caveats: same as Compose.
template <class Arc>
void Intersect(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
               MutableFst<Arc> *ofst,
               const IntersectOptions &opts = IntersectOptions()) {
  using M = Matcher<Fst<Arc>>;
  if (opts.filter_type == AUTO_FILTER) {
    CacheOptions nopts;
    nopts.gc_limit = 0;  // Cache only the last state for fastest copy.
    *ofst = IntersectFst<Arc>(ifst1, ifst2, nopts);
  } else if (opts.filter_type == SEQUENCE_FILTER) {
    IntersectFstOptions<Arc> iopts;
    iopts.gc_limit = 0;  // Cache only the last state for fastest copy.
    *ofst = IntersectFst<Arc>(ifst1, ifst2, iopts);
  } else if (opts.filter_type == ALT_SEQUENCE_FILTER) {
    IntersectFstOptions<Arc, M, AltSequenceComposeFilter<M>> iopts;
    iopts.gc_limit = 0;  // Cache only the last state for fastest copy.
    *ofst = IntersectFst<Arc>(ifst1, ifst2, iopts);
  } else if (opts.filter_type == MATCH_FILTER) {
    IntersectFstOptions<Arc, M, MatchComposeFilter<M>> iopts;
    iopts.gc_limit = 0;  // Cache only the last state for fastest copy.
    *ofst = IntersectFst<Arc>(ifst1, ifst2, iopts);
  }
  if (opts.connect) Connect(ofst);
}

}  // namespace fst

#endif  // FST_INTERSECT_H_
