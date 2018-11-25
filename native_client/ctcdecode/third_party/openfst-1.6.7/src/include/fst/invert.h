// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to invert an FST.

#ifndef FST_INVERT_H_
#define FST_INVERT_H_

#include <fst/arc-map.h>
#include <fst/mutable-fst.h>


namespace fst {

// Mapper to implement inversion of an arc.
template <class A>
struct InvertMapper {
  using FromArc = A;
  using ToArc = A;

  InvertMapper() {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.olabel, arc.ilabel, arc.weight, arc.nextstate);
  }

  constexpr MapFinalAction FinalAction() const {
     return MAP_NO_SUPERFINAL;
  }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return InvertProperties(props);
  }
};

// Inverts the transduction corresponding to an FST by exchanging the
// FST's input and output labels.
//
// Complexity:
//
//   Time: O(V + E)
//   Space: O(1)
//
// where V is the number of states and E is the number of arcs.
template <class Arc>
inline void Invert(const Fst<Arc> &ifst, MutableFst<Arc> *ofst) {
  std::unique_ptr<SymbolTable> input(
      ifst.InputSymbols() ? ifst.InputSymbols()->Copy() : nullptr);
  std::unique_ptr<SymbolTable> output(
      ifst.OutputSymbols() ? ifst.OutputSymbols()->Copy() : nullptr);
  ArcMap(ifst, ofst, InvertMapper<Arc>());
  ofst->SetInputSymbols(output.get());
  ofst->SetOutputSymbols(input.get());
}

// Destructive variant of the above.
template <class Arc>
inline void Invert(MutableFst<Arc> *fst) {
  std::unique_ptr<SymbolTable> input(
      fst->InputSymbols() ? fst->InputSymbols()->Copy() : nullptr);
  std::unique_ptr<SymbolTable> output(
      fst->OutputSymbols() ? fst->OutputSymbols()->Copy() : nullptr);
  ArcMap(fst, InvertMapper<Arc>());
  fst->SetInputSymbols(output.get());
  fst->SetOutputSymbols(input.get());
}

// Inverts the transduction corresponding to an FST by exchanging the
// FST's input and output labels. This version is a delayed FST.
//
// Complexity:
//
//   Time: O(v + e)
//   Space: O(1)
//
// where v is the number of states visited and e is the number of arcs visited.
// Constant time and to visit an input state or arc is assumed and exclusive of
// caching.
template <class A>
class InvertFst : public ArcMapFst<A, A, InvertMapper<A>> {
 public:
  using Arc = A;

  using Mapper = InvertMapper<Arc>;
  using Impl = internal::ArcMapFstImpl<A, A, InvertMapper<A>>;

  explicit InvertFst(const Fst<Arc> &fst)
      : ArcMapFst<Arc, Arc, Mapper>(fst, Mapper()) {
    GetMutableImpl()->SetOutputSymbols(fst.InputSymbols());
    GetMutableImpl()->SetInputSymbols(fst.OutputSymbols());
  }

  // See Fst<>::Copy() for doc.
  InvertFst(const InvertFst<Arc> &fst, bool safe = false)
      : ArcMapFst<Arc, Arc, Mapper>(fst, safe) {}

  // Get a copy of this InvertFst. See Fst<>::Copy() for further doc.
  InvertFst<Arc> *Copy(bool safe = false) const override {
    return new InvertFst(*this, safe);
  }

 private:
  using ImplToFst<Impl>::GetMutableImpl;
};

// Specialization for InvertFst.
template <class Arc>
class StateIterator<InvertFst<Arc>>
    : public StateIterator<ArcMapFst<Arc, Arc, InvertMapper<Arc>>> {
 public:
  explicit StateIterator(const InvertFst<Arc> &fst)
      : StateIterator<ArcMapFst<Arc, Arc, InvertMapper<Arc>>>(fst) {}
};

// Specialization for InvertFst.
template <class Arc>
class ArcIterator<InvertFst<Arc>>
    : public ArcIterator<ArcMapFst<Arc, Arc, InvertMapper<Arc>>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const InvertFst<Arc> &fst, StateId s)
      : ArcIterator<ArcMapFst<Arc, Arc, InvertMapper<Arc>>>(fst, s) {}
};

// Useful alias when using StdArc.
using StdInvertFst = InvertFst<StdArc>;

}  // namespace fst

#endif  // FST_INVERT_H_
