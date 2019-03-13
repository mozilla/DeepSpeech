// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Function that implements epsilon-normalization.

#ifndef FST_EPSNORMALIZE_H_
#define FST_EPSNORMALIZE_H_


#include <fst/arc-map.h>
#include <fst/factor-weight.h>
#include <fst/invert.h>
#include <fst/rmepsilon.h>


namespace fst {

enum EpsNormalizeType { EPS_NORM_INPUT, EPS_NORM_OUTPUT };

// Returns an equivalent FST that is epsilon-normalized. An acceptor is
// epsilon-normalized if it is epsilon-removed. A transducer is input
// epsilon-normalized if additionally if on each path any epsilon input
// label follows all non-epsilon input labels. Output epsilon-normalized
// is defined similarly.
//
// For more information, see:
//
// Mohri, M. 2002. Generic epsilon-removal and input epsilon-normalization
// algorithms for weighted transducers. International Journal of Computer
// Science, 13(1): 129-143, 2002.
template <class Arc>
void EpsNormalize(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                  EpsNormalizeType type = EPS_NORM_INPUT) {
  EpsNormalize<Arc, GALLIC>(ifst, ofst, type);
}

// Same as above, except allows specifying explicitly the gallic weight type.
template <class Arc, GallicType G>
void EpsNormalize(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                  EpsNormalizeType type) {
  VectorFst<GallicArc<Arc, G>> gfst;
  std::unique_ptr<SymbolTable> symbols;
  if (type == EPS_NORM_INPUT) {
    ArcMap(ifst, &gfst, ToGallicMapper<Arc, G>());
    if (ifst.OutputSymbols()) symbols.reset(ifst.OutputSymbols()->Copy());
  } else {  // type == EPS_NORM_OUTPUT
    ArcMap(InvertFst<Arc>(ifst), &gfst, ToGallicMapper<Arc, G>());
    if (ifst.InputSymbols()) symbols.reset(ifst.InputSymbols()->Copy());
  }
  RmEpsilon(&gfst);
  FactorWeightFst<GallicArc<Arc, G>,
                  GallicFactor<typename Arc::Label, typename Arc::Weight, G>>
      fwfst(gfst);
  ArcMap(fwfst, ofst, FromGallicMapper<Arc, G>());
  ofst->SetOutputSymbols(symbols.get());
  if (type == EPS_NORM_OUTPUT) Invert(ofst);
}

}  // namespace fst

#endif  // FST_EPSNORMALIZE_H_
