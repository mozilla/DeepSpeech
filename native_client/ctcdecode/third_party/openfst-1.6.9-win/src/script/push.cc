// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/push.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void Push(MutableFstClass *fst, ReweightType rew_type, float delta,
          bool remove_total_weight) {
  PushArgs1 args(fst, rew_type, delta, remove_total_weight);
  Apply<Operation<PushArgs1>>("Push", fst->ArcType(), &args);
}

void Push(const FstClass &ifst, MutableFstClass *ofst, uint32 flags,
          ReweightType rew_type, float delta) {
  if (!internal::ArcTypesMatch(ifst, *ofst, "Push")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  PushArgs2 args(ifst, ofst, flags, rew_type, delta);
  Apply<Operation<PushArgs2>>("Push", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(Push, StdArc, PushArgs1);
REGISTER_FST_OPERATION(Push, LogArc, PushArgs1);
REGISTER_FST_OPERATION(Push, Log64Arc, PushArgs1);

REGISTER_FST_OPERATION(Push, StdArc, PushArgs2);
REGISTER_FST_OPERATION(Push, LogArc, PushArgs2);
REGISTER_FST_OPERATION(Push, Log64Arc, PushArgs2);

}  // namespace script
}  // namespace fst
