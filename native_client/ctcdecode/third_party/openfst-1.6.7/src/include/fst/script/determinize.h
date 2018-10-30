// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DETERMINIZE_H_
#define FST_SCRIPT_DETERMINIZE_H_

#include <tuple>

#include <fst/determinize.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

struct DeterminizeOptions {
  const float delta;
  const WeightClass &weight_threshold;
  const int64 state_threshold;
  const int64 subsequential_label;
  const DeterminizeType det_type;
  const bool increment_subsequential_label;

  DeterminizeOptions(float delta, const WeightClass &weight_threshold,
                     int64 state_threshold = kNoStateId,
                     int64 subsequential_label = 0,
                     DeterminizeType det_type = DETERMINIZE_FUNCTIONAL,
                     bool increment_subsequential_label = false)
      : delta(delta),
        weight_threshold(weight_threshold),
        state_threshold(state_threshold),
        subsequential_label(subsequential_label),
        det_type(det_type),
        increment_subsequential_label(increment_subsequential_label) {}
};

using DeterminizeArgs = std::tuple<const FstClass &, MutableFstClass *,
                                   const DeterminizeOptions &>;

template <class Arc>
void Determinize(DeterminizeArgs *args) {
  using Weight = typename Arc::Weight;
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  const auto &opts = std::get<2>(*args);
  const auto weight_threshold = *(opts.weight_threshold.GetWeight<Weight>());
  const fst::DeterminizeOptions<Arc> detargs(opts.delta, weight_threshold,
      opts.state_threshold, opts.subsequential_label, opts.det_type,
      opts.increment_subsequential_label);
  Determinize(ifst, ofst, detargs);
}

void Determinize(const FstClass &ifst, MutableFstClass *ofst,
                 const DeterminizeOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DETERMINIZE_H_
