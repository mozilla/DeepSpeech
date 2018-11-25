// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RANDEQUIVALENT_H_
#define FST_SCRIPT_RANDEQUIVALENT_H_

#include <climits>
#include <ctime>

#include <tuple>

#include <fst/randequivalent.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

using RandEquivalentInnerArgs = std::tuple<const FstClass &, const FstClass &,
    int32, float, time_t, const RandGenOptions<RandArcSelection> &>;

using RandEquivalentArgs = WithReturnValue<bool, RandEquivalentInnerArgs>;

template <class Arc>
void RandEquivalent(RandEquivalentArgs *args) {
  const Fst<Arc> &fst1 = *(std::get<0>(args->args).GetFst<Arc>());
  const Fst<Arc> &fst2 = *(std::get<1>(args->args).GetFst<Arc>());
  const auto seed = std::get<4>(args->args);
  const auto &opts = std::get<5>(args->args);
  switch (opts.selector) {
    case UNIFORM_ARC_SELECTOR: {
      const UniformArcSelector<Arc> selector(seed);
      const RandGenOptions<UniformArcSelector<Arc>> ropts(selector,
                                                          opts.max_length);
      args->retval = RandEquivalent(fst1, fst2, std::get<2>(args->args),
                                    std::get<3>(args->args), ropts);
      return;
    }
    case FAST_LOG_PROB_ARC_SELECTOR: {
      const FastLogProbArcSelector<Arc> selector(seed);
      const RandGenOptions<FastLogProbArcSelector<Arc>> ropts(selector,
                                                              opts.max_length);
      args->retval = RandEquivalent(fst1, fst2, std::get<2>(args->args),
                                    std::get<3>(args->args), ropts);
      return;
    }
    case LOG_PROB_ARC_SELECTOR: {
      const LogProbArcSelector<Arc> selector(seed);
      const RandGenOptions<LogProbArcSelector<Arc>> ropts(selector,
                                                          opts.max_length);
      args->retval = RandEquivalent(fst1, fst2, std::get<2>(args->args),
                                    std::get<3>(args->args), ropts);
      return;
    }
  }
}

bool RandEquivalent(const FstClass &fst1, const FstClass &fst2, int32 npath = 1,
    float delta = kDelta, time_t seed = time(nullptr),
    const RandGenOptions<RandArcSelection> &opts =
        RandGenOptions<RandArcSelection>(UNIFORM_ARC_SELECTOR));

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RANDEQUIVALENT_H_
