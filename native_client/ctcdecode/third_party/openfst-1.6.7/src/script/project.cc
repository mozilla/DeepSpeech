// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/project.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {


void Project(MutableFstClass *ofst, ProjectType project_type) {
  ProjectArgs args(ofst, project_type);
  Apply<Operation<ProjectArgs>>("Project", ofst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Project, StdArc, ProjectArgs);
REGISTER_FST_OPERATION(Project, LogArc, ProjectArgs);
REGISTER_FST_OPERATION(Project, Log64Arc, ProjectArgs);

}  // namespace script
}  // namespace fst
