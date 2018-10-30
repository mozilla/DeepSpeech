// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/convert.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

FstClass *Convert(const FstClass &ifst, const string &new_type) {
  ConvertInnerArgs iargs(ifst, new_type);
  ConvertArgs args(iargs);
  Apply<Operation<ConvertArgs>>("Convert", ifst.ArcType(), &args);
  return args.retval;
}

REGISTER_FST_OPERATION(Convert, StdArc, ConvertArgs);
REGISTER_FST_OPERATION(Convert, LogArc, ConvertArgs);
REGISTER_FST_OPERATION(Convert, Log64Arc, ConvertArgs);

}  // namespace script
}  // namespace fst
