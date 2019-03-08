// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_LINEAR_LOGLINEAR_APPLY_H_
#define FST_EXTENSIONS_LINEAR_LOGLINEAR_APPLY_H_

#include <fst/compat.h>
#include <fst/arc.h>
#include <fst/arc-map.h>
#include <fst/compose.h>
#include <fst/determinize.h>
#include <fst/float-weight.h>
#include <fst/fst.h>
#include <fst/minimize.h>
#include <fst/mutable-fst.h>
#include <fst/project.h>
#include <fst/rmepsilon.h>
#include <fst/vector-fst.h>

namespace fst {

// Applies a FST model as a discriminative model to weighted input
// `ifst`. `A` is an arc type with tropical weight of all the
// input/output FSTs.
//
// In general, consider `ifst` an unnormalized probability
// distribution between its input X and output Y, P(X, Y); and `lfst`
// a group of unnormalized probability distributions of all its output
// Z for every input Y, Q(Z|Y). `normalize` controls whether Q is
// normalized for every Y before chaining with P(X, Y). I.e., for a
// path (X, Y, Z) in `ofst` (where Y is hidden),
//
// - When `normalize` is true, its weight is P(X, Y) Q(Z|Y) / sum_z Q(z|Y);
// - When `normalize` is false, its weight is P(X, Y) Q(Z|Y).
template <class A>
void LogLinearApply(const Fst<A> &ifst, const Fst<A> &lfst, MutableFst<A> *ofst,
                    bool normalize = true) {
  LogLinearApply<A, LogArc>(ifst, lfst, ofst, normalize);
}

// This version gives finer control over the arc type (`B`) to be used
// in normalization. `B` is an arc type with log weight (e.g. `LogArc`
// or `Log64Arc`).
template <class A, class B>
void LogLinearApply(const Fst<A> &ifst, const Fst<A> &lfst, MutableFst<A> *ofst,
                    bool normalize = true) {
  if (normalize) {
    VectorFst<A> unnormalized_ofst, rescored_ifsa;
    Compose(ifst, lfst, &unnormalized_ofst);
    {
      VectorFst<A> tropical_ifsa(unnormalized_ofst);
      Project(&tropical_ifsa, PROJECT_INPUT);
      {
        VectorFst<B> minimal_log_ifsa;
        {
          VectorFst<B> log_ifsa;
          ArcMap(tropical_ifsa, &log_ifsa, WeightConvertMapper<A, B>());
          RmEpsilon(&log_ifsa);
          Determinize(log_ifsa, &minimal_log_ifsa);
        }
        Minimize(&minimal_log_ifsa);
        ArcMap(&minimal_log_ifsa, InvertWeightMapper<B>());
        ArcMap(minimal_log_ifsa, &tropical_ifsa, WeightConvertMapper<B, A>());
      }
      ArcSort(&tropical_ifsa, OLabelCompare<A>());
      Compose(tropical_ifsa, ifst, &rescored_ifsa);
    }
    ArcSort(&rescored_ifsa, OLabelCompare<A>());
    Compose(rescored_ifsa, unnormalized_ofst, ofst);
  } else {
    Compose(ifst, lfst, ofst);
  }
}

}  // namespace fst

#endif  // FST_EXTENSIONS_LINEAR_LOGLINEAR_APPLY_H_
