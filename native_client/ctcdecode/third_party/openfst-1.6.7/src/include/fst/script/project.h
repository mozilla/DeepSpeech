// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PROJECT_H_
#define FST_SCRIPT_PROJECT_H_

#include <utility>

#include <fst/project.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ProjectArgs = std::pair<MutableFstClass *, ProjectType>;

template <class Arc>
void Project(ProjectArgs *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  Project(fst, std::get<1>(*args));
}

void Project(MutableFstClass *fst, ProjectType project_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PROJECT_H_
