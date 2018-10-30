// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// The FST script interface permits users to interact with FSTs without knowing
// their arc type. It does this by mapping compile-time polymorphism (in the
// form of a arc-templated FST types) onto a shared virtual interface. It also
// supports arc extension via a DSO interface. Due to the overhead of virtual
// dispatch and registered function lookups, the script API is somewhat slower
// then library API provided by types like StdVectorFst, but has the advantage
// that it is designed not to crash (and to provide useful debugging
// information) upon common user errors like passing invalid indices or
// attempting comparison of incompatible FSTs. It is used both by the FST
// binaries and the Python extension.
//
// This header includes all of the FST script functionality.

#ifndef FST_SCRIPT_FSTSCRIPT_H_
#define FST_SCRIPT_FSTSCRIPT_H_

// Major classes
#include <fst/script/arciterator-class.h>
#include <fst/script/encodemapper-class.h>
#include <fst/script/fst-class.h>
#include <fst/script/stateiterator-class.h>
#include <fst/script/text-io.h>
#include <fst/script/weight-class.h>

// Flag-to-enum parsers.
#include <fst/script/getters.h>
// Templates like Operation<> and Apply<>.
#include <fst/script/script-impl.h>

// Operations.
#include <fst/script/arcsort.h>
#include <fst/script/closure.h>
#include <fst/script/compile.h>
#include <fst/script/compose.h>
#include <fst/script/concat.h>
#include <fst/script/connect.h>
#include <fst/script/convert.h>
#include <fst/script/decode.h>
#include <fst/script/determinize.h>
#include <fst/script/difference.h>
#include <fst/script/disambiguate.h>
#include <fst/script/draw.h>
#include <fst/script/encode.h>
#include <fst/script/epsnormalize.h>
#include <fst/script/equal.h>
#include <fst/script/equivalent.h>
#include <fst/script/info.h>
#include <fst/script/intersect.h>
#include <fst/script/invert.h>
#include <fst/script/isomorphic.h>
#include <fst/script/map.h>
#include <fst/script/minimize.h>
#include <fst/script/print.h>
#include <fst/script/project.h>
#include <fst/script/prune.h>
#include <fst/script/push.h>
#include <fst/script/randequivalent.h>
#include <fst/script/randgen.h>
#include <fst/script/relabel.h>
#include <fst/script/replace.h>
#include <fst/script/reverse.h>
#include <fst/script/reweight.h>
#include <fst/script/rmepsilon.h>
#include <fst/script/shortest-distance.h>
#include <fst/script/shortest-path.h>
#include <fst/script/synchronize.h>
#include <fst/script/topsort.h>
#include <fst/script/union.h>
#include <fst/script/verify.h>

// This class is necessary because registering each of the operations
// separately overfills the stack, as there's so many of them.
namespace fst {
namespace script {

template <class Arc>
class AllFstOperationsRegisterer {
 public:
  AllFstOperationsRegisterer() {
    RegisterBatch1();
    RegisterBatch2();
  }

 private:
  void RegisterBatch1() {
    REGISTER_FST_OPERATION(ArcSort, Arc, ArcSortArgs);
    REGISTER_FST_OPERATION(Closure, Arc, ClosureArgs);
    REGISTER_FST_OPERATION(CompileFstInternal, Arc, CompileFstArgs);
    REGISTER_FST_OPERATION(Compose, Arc, ComposeArgs);
    REGISTER_FST_OPERATION(Concat, Arc, ConcatArgs1);
    REGISTER_FST_OPERATION(Concat, Arc, ConcatArgs2);
    REGISTER_FST_OPERATION(Connect, Arc, MutableFstClass);
    REGISTER_FST_OPERATION(Convert, Arc, ConvertArgs);
    REGISTER_FST_OPERATION(Decode, Arc, DecodeArgs1);
    REGISTER_FST_OPERATION(Decode, Arc, DecodeArgs2);
    REGISTER_FST_OPERATION(Determinize, Arc, DeterminizeArgs);
    REGISTER_FST_OPERATION(Difference, Arc, DifferenceArgs);
    REGISTER_FST_OPERATION(Disambiguate, Arc, DisambiguateArgs);
    REGISTER_FST_OPERATION(DrawFst, Arc, FstDrawerArgs);
    REGISTER_FST_OPERATION(Encode, Arc, EncodeArgs1);
    REGISTER_FST_OPERATION(Encode, Arc, EncodeArgs2);
    REGISTER_FST_OPERATION(EpsNormalize, Arc, EpsNormalizeArgs);
    REGISTER_FST_OPERATION(Equal, Arc, EqualArgs);
    REGISTER_FST_OPERATION(Equivalent, Arc, EquivalentArgs);
    REGISTER_FST_OPERATION(PrintFstInfo, Arc, InfoArgs);
    REGISTER_FST_OPERATION(GetFstInfo, Arc, GetInfoArgs);
    REGISTER_FST_OPERATION(InitArcIteratorClass, Arc,
                           InitArcIteratorClassArgs);
    REGISTER_FST_OPERATION(InitEncodeMapperClass, Arc,
                           InitEncodeMapperClassArgs);
    REGISTER_FST_OPERATION(InitMutableArcIteratorClass, Arc,
                           InitMutableArcIteratorClassArgs);
    REGISTER_FST_OPERATION(InitStateIteratorClass, Arc,
                           InitStateIteratorClassArgs);
  }

  void RegisterBatch2() {
    REGISTER_FST_OPERATION(Intersect, Arc, IntersectArgs);
    REGISTER_FST_OPERATION(Invert, Arc, MutableFstClass);
    REGISTER_FST_OPERATION(Map, Arc, MapArgs);
    REGISTER_FST_OPERATION(Minimize, Arc, MinimizeArgs);
    REGISTER_FST_OPERATION(PrintFst, Arc, FstPrinterArgs);
    REGISTER_FST_OPERATION(Project, Arc, ProjectArgs);
    REGISTER_FST_OPERATION(Prune, Arc, PruneArgs1);
    REGISTER_FST_OPERATION(Prune, Arc, PruneArgs2);
    REGISTER_FST_OPERATION(Push, Arc, PushArgs1);
    REGISTER_FST_OPERATION(Push, Arc, PushArgs2);
    REGISTER_FST_OPERATION(RandEquivalent, Arc, RandEquivalentArgs);
    REGISTER_FST_OPERATION(RandGen, Arc, RandGenArgs);
    REGISTER_FST_OPERATION(Relabel, Arc, RelabelArgs1);
    REGISTER_FST_OPERATION(Relabel, Arc, RelabelArgs2);
    REGISTER_FST_OPERATION(Replace, Arc, ReplaceArgs);
    REGISTER_FST_OPERATION(Reverse, Arc, ReverseArgs);
    REGISTER_FST_OPERATION(Reweight, Arc, ReweightArgs);
    REGISTER_FST_OPERATION(RmEpsilon, Arc, RmEpsilonArgs);
    REGISTER_FST_OPERATION(ShortestDistance, Arc, ShortestDistanceArgs1);
    REGISTER_FST_OPERATION(ShortestDistance, Arc, ShortestDistanceArgs2);
    REGISTER_FST_OPERATION(ShortestPath, Arc, ShortestPathArgs);
    REGISTER_FST_OPERATION(Synchronize, Arc, SynchronizeArgs);
    REGISTER_FST_OPERATION(TopSort, Arc, TopSortArgs);
    REGISTER_FST_OPERATION(Union, Arc, UnionArgs);
    REGISTER_FST_OPERATION(Verify, Arc, VerifyArgs);
  }
};

}  // namespace script
}  // namespace fst

#define REGISTER_FST_OPERATIONS(Arc) \
  AllFstOperationsRegisterer<Arc> register_all_fst_operations##Arc;

#endif  // FST_SCRIPT_FSTSCRIPT_H_
