// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PUSH_H_
#define FST_SCRIPT_PUSH_H_

#include <tuple>

#include <fst/push.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using PushArgs1 = std::tuple<MutableFstClass *, ReweightType, float, bool>;

template <class Arc>
void Push(PushArgs1 *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  Push(fst, std::get<1>(*args), std::get<2>(*args), std::get<3>(*args));
}

using PushArgs2 = std::tuple<const FstClass &, MutableFstClass *, uint32,
                             ReweightType, float>;

template <class Arc>
void Push(PushArgs2 *args) {
  const Fst<Arc> &ifst = *(std::get<0>(*args).GetFst<Arc>());
  MutableFst<Arc> *ofst = std::get<1>(*args)->GetMutableFst<Arc>();
  switch (std::get<3>(*args)) {
    case REWEIGHT_TO_FINAL: {
      Push<Arc, REWEIGHT_TO_FINAL>(ifst, ofst, std::get<2>(*args),
                                   std::get<4>(*args));
      return;
    }
    case REWEIGHT_TO_INITIAL: {
      Push<Arc, REWEIGHT_TO_INITIAL>(ifst, ofst, std::get<2>(*args),
                                     std::get<4>(*args));
      return;
    }
  }
}

void Push(MutableFstClass *fst, ReweightType rew_type, float delta = kDelta,
          bool remove_total_weight = false);

void Push(const FstClass &ifst, MutableFstClass *ofst, uint32 flags,
          ReweightType rew_type, float delta = kDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PUSH_H_
