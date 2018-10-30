// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <string>

#include <fst/script/fst-class.h>
#include <fst/script/info.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void PrintFstInfo(const FstClass &fst, bool test_properties,
                  const string &arc_filter, const string &info_type, bool pipe,
                  bool verify) {
  InfoArgs args(fst, test_properties, arc_filter, info_type, pipe, verify);
  Apply<Operation<InfoArgs>>("PrintFstInfo", fst.ArcType(), &args);
}

void GetFstInfo(const FstClass &fst, bool test_properties,
                const string &arc_filter, const string &info_type, bool verify,
                FstInfo *result) {
  GetInfoArgs args(fst, test_properties, arc_filter, info_type, verify, result);
  Apply<Operation<GetInfoArgs>>("GetFstInfo", fst.ArcType(), &args);
}

REGISTER_FST_OPERATION(PrintFstInfo, StdArc, InfoArgs);
REGISTER_FST_OPERATION(PrintFstInfo, LogArc, InfoArgs);
REGISTER_FST_OPERATION(PrintFstInfo, Log64Arc, InfoArgs);

REGISTER_FST_OPERATION(GetFstInfo, StdArc, GetInfoArgs);
REGISTER_FST_OPERATION(GetFstInfo, LogArc, GetInfoArgs);
REGISTER_FST_OPERATION(GetFstInfo, Log64Arc, GetInfoArgs);

}  // namespace script
}  // namespace fst
