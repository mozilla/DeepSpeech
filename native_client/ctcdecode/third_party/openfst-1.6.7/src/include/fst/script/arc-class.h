// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ARC_CLASS_H_
#define FST_SCRIPT_ARC_CLASS_H_

#include <fst/script/weight-class.h>

namespace fst {
namespace script {

// A struct representing an arc while ignoring arc type. It is passed as an
// argument to AddArc.

struct ArcClass {
  template <class Arc>
  explicit ArcClass(const Arc &arc)
      : ilabel(arc.ilabel), olabel(arc.olabel), weight(arc.weight),
        nextstate(arc.nextstate) {}

  ArcClass(int64 ilabel, int64 olabel, const WeightClass &weight,
           int64 nextstate)
      : ilabel(ilabel), olabel(olabel), weight(weight), nextstate(nextstate) {}

  template <class Arc>
  Arc GetArc() const {
    return Arc(ilabel, olabel, *(weight.GetWeight<typename Arc::Weight>()),
               nextstate);
  }

  int64 ilabel;
  int64 olabel;
  WeightClass weight;
  int64 nextstate;
};

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ARC_CLASS_H_
