// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CONVERT_H_
#define FST_SCRIPT_CONVERT_H_

#include <memory>
#include <string>
#include <utility>

#include <fst/register.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ConvertInnerArgs = std::pair<const FstClass &, const string &>;

using ConvertArgs = WithReturnValue<FstClass *, ConvertInnerArgs>;

template <class Arc>
void Convert(ConvertArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(args->args).GetFst<Arc>());
  const string &new_type = std::get<1>(args->args);
  std::unique_ptr<Fst<Arc>> result(Convert(fst, new_type));
  args->retval = result ? new FstClass(*result) : nullptr;
}

FstClass *Convert(const FstClass &fst, const string &new_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CONVERT_H_
