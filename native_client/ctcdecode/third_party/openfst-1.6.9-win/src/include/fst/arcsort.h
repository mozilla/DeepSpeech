// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to sort arcs in an FST.

#ifndef FST_ARCSORT_H_
#define FST_ARCSORT_H_

#include <algorithm>
#include <string>
#include <vector>

#include <fst/cache.h>
#include <fst/state-map.h>
#include <fst/test-properties.h>


namespace fst {

template <class Arc, class Compare>
class ArcSortMapper {
 public:
  using FromArc = Arc;
  using ToArc = Arc;

  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  ArcSortMapper(const Fst<Arc> &fst, const Compare &comp)
      : fst_(fst), comp_(comp), i_(0) {}

  // Allows updating Fst argument; pass only if changed.
  ArcSortMapper(const ArcSortMapper<Arc, Compare> &mapper,
                const Fst<Arc> *fst = nullptr)
      : fst_(fst ? *fst : mapper.fst_), comp_(mapper.comp_), i_(0) {}

  StateId Start() { return fst_.Start(); }

  Weight Final(StateId s) const { return fst_.Final(s); }

  void SetState(StateId s) {
    i_ = 0;
    arcs_.clear();
    arcs_.reserve(fst_.NumArcs(s));
    for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
      arcs_.push_back(aiter.Value());
    }
    std::sort(arcs_.begin(), arcs_.end(), comp_);
  }

  bool Done() const { return i_ >= arcs_.size(); }

  const Arc &Value() const { return arcs_[i_]; }

  void Next() { ++i_; }

  MapSymbolsAction InputSymbolsAction() const { return MAP_COPY_SYMBOLS; }

  MapSymbolsAction OutputSymbolsAction() const { return MAP_COPY_SYMBOLS; }

  uint64_t Properties(uint64_t props) const { return comp_.Properties(props); }

 private:
  const Fst<Arc> &fst_;
  const Compare &comp_;
  std::vector<Arc> arcs_;
  std::ptrdiff_t i_;  // current arc position

  ArcSortMapper &operator=(const ArcSortMapper &) = delete;
};

// Sorts the arcs in an FST according to function object 'comp' of type Compare.
// This version modifies its input. Comparison function objects ILabelCompare
// and OLabelCompare are provided by the library. In general, Compare must meet
// the requirements for a  comparison function object (e.g., similar to those
// used by std::sort). It must also have a member Properties(uint64_t) that
// specifies the known properties of the sorted FST; it takes as argument the
// input FST's known properties before the sort.
//
// Complexity:
//
// - Time: O(v d log d)
// - Space: O(d)
//
// where v = # of states and d = maximum out-degree.
template <class Arc, class Compare>
void ArcSort(MutableFst<Arc> *fst, Compare comp) {
  ArcSortMapper<Arc, Compare> mapper(*fst, comp);
  StateMap(fst, mapper);
}

using ArcSortFstOptions = CacheOptions;

// Sorts the arcs in an FST according to function object 'comp' of type Compare.
// This version is a delayed FST. Comparsion function objects ILabelCompare and
// OLabelCompare are provided by the library. In general, Compare must meet the
// requirements for a comparision function object (e.g., similar to those
// used by std::sort). It must also have a member Properties(uint64_t) that
// specifies the known properties of the sorted FST; it takes as argument the
// input FST's known properties.
//
// Complexity:
//
// - Time: O(v d log d)
// - Space: O(d)
//
// where v = # of states visited, d = maximum out-degree of states visited.
// Constant time and space to visit an input state is assumed and exclusive of
// caching.
template <class Arc, class Compare>
class ArcSortFst : public StateMapFst<Arc, Arc, ArcSortMapper<Arc, Compare>> {
  using StateMapFst<Arc, Arc, ArcSortMapper<Arc, Compare>>::GetImpl;

 public:
  using StateId = typename Arc::StateId;
  using Mapper = ArcSortMapper<Arc, Compare>;

  ArcSortFst(const Fst<Arc> &fst, const Compare &comp)
      : StateMapFst<Arc, Arc, Mapper>(fst,
                                      ArcSortMapper<Arc, Compare>(fst, comp)) {}

  ArcSortFst(const Fst<Arc> &fst, const Compare &comp,
             const ArcSortFstOptions &opts)
      : StateMapFst<Arc, Arc, Mapper>(fst, Mapper(fst, comp), opts) {}

  // See Fst<>::Copy() for doc.
  ArcSortFst(const ArcSortFst<Arc, Compare> &fst, bool safe = false)
      : StateMapFst<Arc, Arc, Mapper>(fst, safe) {}

  // Gets a copy of this ArcSortFst. See Fst<>::Copy() for further doc.
  ArcSortFst<Arc, Compare> *Copy(bool safe = false) const override {
    return new ArcSortFst(*this, safe);
  }

  size_t NumArcs(StateId s) const override {
    return GetImpl()->GetFst()->NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) const override {
    return GetImpl()->GetFst()->NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) const override {
    return GetImpl()->GetFst()->NumOutputEpsilons(s);
  }
};

// Specialization for ArcSortFst.
template <class Arc, class Compare>
class StateIterator<ArcSortFst<Arc, Compare>>
    : public StateIterator<StateMapFst<Arc, Arc, ArcSortMapper<Arc, Compare>>> {
 public:
  explicit StateIterator(const ArcSortFst<Arc, Compare> &fst)
      : StateIterator<StateMapFst<Arc, Arc, ArcSortMapper<Arc, Compare>>>(fst) {
  }
};

// Specialization for ArcSortFst.
template <class Arc, class Compare>
class ArcIterator<ArcSortFst<Arc, Compare>>
    : public ArcIterator<StateMapFst<Arc, Arc, ArcSortMapper<Arc, Compare>>> {
 public:
  ArcIterator(const ArcSortFst<Arc, Compare> &fst, typename Arc::StateId s)
      : ArcIterator<StateMapFst<Arc, Arc, ArcSortMapper<Arc, Compare>>>(fst,
                                                                        s) {}
};

// Compare class for comparing input labels of arcs.
template <class Arc>
class ILabelCompare {
 public:
  ILabelCompare() {}

  bool operator()(const Arc &arc1, const Arc &arc2) const {
    return arc1.ilabel < arc2.ilabel;
  }

  uint64_t Properties(uint64_t props) const {
    return (props & kArcSortProperties) | kILabelSorted |
           (props & kAcceptor ? kOLabelSorted : 0);
  }
};

// Compare class for comparing output labels of arcs.
template <class Arc>
class OLabelCompare {
 public:
  OLabelCompare() {}

  bool operator()(const Arc &arc1, const Arc &arc2) const {
    return arc1.olabel < arc2.olabel;
  }

  uint64_t Properties(uint64_t props) const {
    return (props & kArcSortProperties) | kOLabelSorted |
           (props & kAcceptor ? kILabelSorted : 0);
  }
};

// Useful aliases when using StdArc.

template <class Compare>
using StdArcSortFst = ArcSortFst<StdArc, Compare>;

using StdILabelCompare = ILabelCompare<StdArc>;

using StdOLabelCompare = OLabelCompare<StdArc>;

}  // namespace fst

#endif  // FST_ARCSORT_H_
