// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DISAMBIGUATE_H_
#define FST_SCRIPT_DISAMBIGUATE_H_

#include <tuple>
#include <utility>

#include <fst/disambiguate.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

struct DisambiguateOptions {
  const float delta;
  const WeightClass &weight_threshold;
  const int64 state_threshold;
  const int64 subsequential_label;

  DisambiguateOptions(float delta, const WeightClass &weight_threshold,
                      int64 state_threshold = kNoStateId,
                      int64 subsequential_label = 0)
      : delta(delta),
        weight_threshold(weight_threshold),
        state_threshold(state_threshold),
        subsequential_label(subsequential_label) {}
};

using DisambiguateArgs = std::tuple<const FstClass &, MutableFstClass *,
                                     const DisambiguateOptions &>;

template <class Arc>
void Disambiguate(DisambiguateArgs *args) {
  using Weight = typename Arc::Weight;
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  const auto &opts = std::get<2>(*args);
  const auto weight_threshold = *(opts.weight_threshold.GetWeight<Weight>());
  const fst::DisambiguateOptions<Arc> disargs(opts.delta, weight_threshold,
                                                  opts.state_threshold,
                                                  opts.subsequential_label);
  Disambiguate(ifst, ofst, disargs);
}

void Disambiguate(const FstClass &ifst, MutableFstClass *ofst,
                  const DisambiguateOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DISAMBIGUATE_H_
