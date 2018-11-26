// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_MINIMIZE_H_
#define FST_SCRIPT_MINIMIZE_H_

#include <tuple>

#include <fst/minimize.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using MinimizeArgs = std::tuple<MutableFstClass *, MutableFstClass *, float,
                                bool>;

template <class Arc>
void Minimize(MinimizeArgs *args) {
  MutableFst<Arc> *ofst1 = std::get<0>(*args)->GetMutableFst<Arc>();
  MutableFst<Arc> *ofst2 = (std::get<1>(*args) ?
                            std::get<1>(*args)->GetMutableFst<Arc>() :
                            nullptr);
  Minimize(ofst1, ofst2, std::get<2>(*args), std::get<3>(*args));
}

void Minimize(MutableFstClass *ofst1, MutableFstClass *ofst2 = nullptr,
              float delta = kShortestDelta, bool allow_nondet = false);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_MINIMIZE_H_
