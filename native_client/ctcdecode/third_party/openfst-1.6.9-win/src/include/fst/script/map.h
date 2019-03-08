// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_MAP_H_
#define FST_SCRIPT_MAP_H_

#include <memory>
#include <tuple>

#include <fst/arc-map.h>
#include <fst/state-map.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

template <class M>
Fst<typename M::ToArc> *ArcMap(const Fst<typename M::FromArc> &fst,
                               const M &mapper) {
  using ToArc = typename M::ToArc;
  auto *ofst = new VectorFst<ToArc>;
  ArcMap(fst, ofst, mapper);
  return ofst;
}

template <class M>
Fst<typename M::ToArc> *StateMap(const Fst<typename M::FromArc> &fst,
                                 const M &mapper) {
  using ToArc = typename M::ToArc;
  auto *ofst = new VectorFst<ToArc>;
  StateMap(fst, ofst, mapper);
  return ofst;
}

enum MapType {
  ARC_SUM_MAPPER,
  ARC_UNIQUE_MAPPER,
  IDENTITY_MAPPER,
  INPUT_EPSILON_MAPPER,
  INVERT_MAPPER,
  OUTPUT_EPSILON_MAPPER,
  PLUS_MAPPER,
  POWER_MAPPER,
  QUANTIZE_MAPPER,
  RMWEIGHT_MAPPER,
  SUPERFINAL_MAPPER,
  TIMES_MAPPER,
  TO_LOG_MAPPER,
  TO_LOG64_MAPPER,
  TO_STD_MAPPER
};

using MapInnerArgs =
    std::tuple<const FstClass &, MapType, float, double, const WeightClass &>;

using MapArgs = WithReturnValue<FstClass *, MapInnerArgs>;

template <class Arc>
void Map(MapArgs *args) {
  using Weight = typename Arc::Weight;
  const Fst<Arc> &ifst = *(std::get<0>(args->args).GetFst<Arc>());
  const auto map_type = std::get<1>(args->args);
  switch (map_type) {
    case ARC_SUM_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(StateMap(ifst, ArcSumMapper<Arc>(ifst)));
      args->retval = new FstClass(*ofst);
      return;
    }
    case ARC_UNIQUE_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(
          StateMap(ifst, ArcUniqueMapper<Arc>(ifst)));
      args->retval = new FstClass(*ofst);
      return;
    }
    case IDENTITY_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, IdentityArcMapper<Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case INPUT_EPSILON_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, InputEpsilonMapper<Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case INVERT_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, InvertWeightMapper<Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case OUTPUT_EPSILON_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, OutputEpsilonMapper<Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case PLUS_MAPPER: {
      const auto weight = *(std::get<4>(args->args).GetWeight<Weight>());
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, PlusMapper<Arc>(weight)));
      args->retval = new FstClass(*ofst);
      return;
    }
    case POWER_MAPPER: {
      const auto power = std::get<3>(args->args);
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, PowerMapper<Arc>(power)));
      args->retval = new FstClass(*ofst);
      return;
    }
    case QUANTIZE_MAPPER: {
      const auto delta = std::get<2>(args->args);
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, QuantizeMapper<Arc>(delta)));
      args->retval = new FstClass(*ofst);
      return;
    }
    case RMWEIGHT_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, RmWeightMapper<Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case SUPERFINAL_MAPPER: {
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, SuperFinalMapper<Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case TIMES_MAPPER: {
      const auto weight = *(std::get<4>(args->args).GetWeight<Weight>());
      std::unique_ptr<Fst<Arc>> ofst(ArcMap(ifst, TimesMapper<Arc>(weight)));
      args->retval = new FstClass(*ofst);
      return;
    }
    case TO_LOG_MAPPER: {
      std::unique_ptr<Fst<LogArc>> ofst(
          ArcMap(ifst, WeightConvertMapper<Arc, LogArc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case TO_LOG64_MAPPER: {
      std::unique_ptr<Fst<Log64Arc>> ofst(
          ArcMap(ifst, WeightConvertMapper<Arc, Log64Arc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
    case TO_STD_MAPPER: {
      std::unique_ptr<Fst<StdArc>> ofst(
          ArcMap(ifst, WeightConvertMapper<Arc, StdArc>()));
      args->retval = new FstClass(*ofst);
      return;
    }
  }
}

FstClass *Map(const FstClass &ifst, MapType map_type, float delta, double power,
              const WeightClass &weight);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_MAP_H_
