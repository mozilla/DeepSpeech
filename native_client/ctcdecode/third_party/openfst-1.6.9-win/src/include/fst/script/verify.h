// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_VERIFY_H_
#define FST_SCRIPT_VERIFY_H_

#include <fst/verify.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using VerifyArgs = WithReturnValue<bool, const FstClass &>;

template <class Arc>
void Verify(VerifyArgs *args) {
  const Fst<Arc> &fst = *(args->args.GetFst<Arc>());
  args->retval = Verify(fst);
}

bool Verify(const FstClass &fst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_VERIFY_H_
