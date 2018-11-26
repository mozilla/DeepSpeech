// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_INFO_H_
#define FST_SCRIPT_INFO_H_

#include <string>
#include <tuple>

#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/info-impl.h>

namespace fst {
namespace script {

using InfoArgs = std::tuple<const FstClass &, bool, const string &,
                            const string &, bool, bool>;

template <class Arc>
void PrintFstInfo(InfoArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  const FstInfo fstinfo(fst, std::get<1>(*args), std::get<2>(*args),
                        std::get<3>(*args), std::get<4>(*args));
  PrintFstInfoImpl(fstinfo, std::get<5>(*args));
  if (std::get<5>(*args)) fst.Write("");
}

void PrintFstInfo(const FstClass &f, bool test_properties,
                  const string &arc_filter, const string &info_type, bool pipe,
                  bool verify);

using GetInfoArgs = std::tuple<const FstClass &, bool, const string &,
                               const string &, bool, FstInfo *>;

template <class Arc>
void GetFstInfo(GetInfoArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  *(std::get<5>(*args)) = FstInfo(fst, std::get<1>(*args), std::get<2>(*args),
                                  std::get<3>(*args), std::get<4>(*args));
}

void GetFstInfo(const FstClass &fst, bool test_properties,
                const string &arc_filter, const string &info_type, bool verify,
                FstInfo *info);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_INFO_H_
