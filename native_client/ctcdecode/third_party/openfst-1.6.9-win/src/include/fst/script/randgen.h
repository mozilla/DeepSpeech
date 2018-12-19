// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RANDGEN_H_
#define FST_SCRIPT_RANDGEN_H_

#include <ctime>

#include <tuple>

#include <fst/randgen.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

using RandGenArgs = std::tuple<const FstClass &, MutableFstClass *, time_t,
                               const RandGenOptions<RandArcSelection> &>;

template <class Arc>
void RandGen(RandGenArgs *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  const time_t seed = std::get<2>(*args);
  const auto &opts = std::get<3>(*args);
  switch (opts.selector) {
    case UNIFORM_ARC_SELECTOR: {
      const UniformArcSelector<Arc> selector(seed);
      const RandGenOptions<UniformArcSelector<Arc>> ropts(
          selector, opts.max_length, opts.npath, opts.weighted,
          opts.remove_total_weight);
      RandGen(ifst, ofst, ropts);
      return;
    }
    case FAST_LOG_PROB_ARC_SELECTOR: {
      const FastLogProbArcSelector<Arc> selector(seed);
      const RandGenOptions<FastLogProbArcSelector<Arc>> ropts(
          selector, opts.max_length, opts.npath, opts.weighted,
          opts.remove_total_weight);
      RandGen(ifst, ofst, ropts);
      return;
    }
    case LOG_PROB_ARC_SELECTOR: {
      const LogProbArcSelector<Arc> selector(seed);
      const RandGenOptions<LogProbArcSelector<Arc>> ropts(
          selector, opts.max_length, opts.npath, opts.weighted,
          opts.remove_total_weight);
      RandGen(ifst, ofst, ropts);
      return;
    }
  }
}

void RandGen(const FstClass &ifst, MutableFstClass *ofst,
             time_t seed = time(nullptr),
             const RandGenOptions<RandArcSelection> &opts =
                 RandGenOptions<RandArcSelection>(UNIFORM_ARC_SELECTOR));

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RANDGEN_H_
