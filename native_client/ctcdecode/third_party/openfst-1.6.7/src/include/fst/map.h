// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Compatibility file for old-style Map() functions and MapFst class that have
// been renamed to ArcMap (cf. StateMap).

#ifndef FST_MAP_H_
#define FST_MAP_H_


#include <fst/arc-map.h>


namespace fst {

template <class A, class C>
void Map(MutableFst<A> *fst, C *mapper) {
  ArcMap(fst, mapper);
}

template <class A, class C>
void Map(MutableFst<A> *fst, C mapper) {
  ArcMap(fst, mapper);
}

template <class A, class B, class C>
void Map(const Fst<A> &ifst, MutableFst<B> *ofst, C *mapper) {
  ArcMap(ifst, ofst, mapper);
}

template <class A, class B, class C>
void Map(const Fst<A> &ifst, MutableFst<B> *ofst, C mapper) {
  ArcMap(ifst, ofst, mapper);
}

using MapFstOptions = ArcMapFstOptions;

template <class A, class B, class C>
class MapFst : public ArcMapFst<A, B, C> {
 public:
  using FromArc = A;
  using ToArc = B;

  using StateId = typename ToArc::StateId;
  using Weight = typename ToArc::Weight;

  using State = CacheState<B>;

  MapFst(const Fst<A> &fst, const C &mapper, const MapFstOptions &opts)
      : ArcMapFst<A, B, C>(fst, mapper, opts) {}

  MapFst(const Fst<A> &fst, C *mapper, const MapFstOptions &opts)
      : ArcMapFst<A, B, C>(fst, mapper, opts) {}

  MapFst(const Fst<A> &fst, const C &mapper)
      : ArcMapFst<A, B, C>(fst, mapper) {}

  MapFst(const Fst<A> &fst, C *mapper) : ArcMapFst<A, B, C>(fst, mapper) {}

  // See Fst<>::Copy() for doc.
  MapFst(const MapFst<A, B, C> &fst, bool safe = false)
      : ArcMapFst<A, B, C>(fst, safe) {}

  // Get a copy of this MapFst. See Fst<>::Copy() for further doc.
  MapFst<A, B, C> *Copy(bool safe = false) const override {
    return new MapFst(*this, safe);
  }
};

// Specialization for MapFst.
template <class A, class B, class C>
class StateIterator<MapFst<A, B, C>>
    : public StateIterator<ArcMapFst<A, B, C>> {
 public:
  explicit StateIterator(const ArcMapFst<A, B, C> &fst)
      : StateIterator<ArcMapFst<A, B, C>>(fst) {}
};

// Specialization for MapFst.
template <class A, class B, class C>
class ArcIterator<MapFst<A, B, C>> : public ArcIterator<ArcMapFst<A, B, C>> {
 public:
  ArcIterator(const ArcMapFst<A, B, C> &fst, typename A::StateId s)
      : ArcIterator<ArcMapFst<A, B, C>>(fst, s) {}
};

// For backwards compatibility only; use IdentityArcMapper otherwise.
template <class A>
struct IdentityMapper {
  using FromArc = A;
  using ToArc = A;

  ToArc operator()(const FromArc &arc) const { return arc; }

  constexpr MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }
};

}  // namespace fst

#endif  // FST_MAP_H_
