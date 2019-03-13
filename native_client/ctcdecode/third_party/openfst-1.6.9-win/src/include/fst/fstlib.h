// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// This is a library for constructing, combining, optimizing, and searching
// "weighted finite-state transducers" (FSTs). Weighted finite-state transducers
// are automata where each transition has an input label, an output label, and a
// weight. The more familiar finite-state acceptor is represented as a
// transducer with each transition's input and output the same. Finite-state
// acceptors are used to represent sets of strings (specifically, "regular" or
// "rational sets"); finite-state transducers are used to represent binary
// relations between pairs of strings (specifically, "rational transductions").
// The weights can be used to represent the cost of taking a particular
// transition.
//
// In this library, transducers are templated on the Arc (transition)
// definition, which allows changing the label, weight, and state ID sets.
// Labels and state IDs are restricted to signed integral types but the weight
// can be an arbitrary type whose members satisfy certain algebraic ("semiring")
// properties.
//
// This convenience file includes all other FST header files.

#ifndef FST_FSTLIB_H_
#define FST_FSTLIB_H_


// Abstract FST classes.
#include <fst/expanded-fst.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>

// Concrete FST classes.
#include <fst/compact-fst.h>
#include <fst/const-fst.h>
#include <fst/edit-fst.h>
#include <fst/vector-fst.h>

// FST algorithms and delayed FST classes.
#include <fst/arc-map.h>
#include <fst/arcsort.h>
#include <fst/closure.h>
#include <fst/compose.h>
#include <fst/concat.h>
#include <fst/connect.h>
#include <fst/determinize.h>
#include <fst/difference.h>
#include <fst/disambiguate.h>
#include <fst/encode.h>
#include <fst/epsnormalize.h>
#include <fst/equal.h>
#include <fst/equivalent.h>
#include <fst/factor-weight.h>
#include <fst/intersect.h>
#include <fst/invert.h>
#include <fst/isomorphic.h>
#include <fst/map.h>
#include <fst/minimize.h>
#include <fst/project.h>
#include <fst/prune.h>
#include <fst/push.h>
#include <fst/randequivalent.h>
#include <fst/randgen.h>
#include <fst/rational.h>
#include <fst/relabel.h>
#include <fst/replace.h>
#include <fst/replace-util.h>
#include <fst/reverse.h>
#include <fst/reweight.h>
#include <fst/rmepsilon.h>
#include <fst/rmfinalepsilon.h>
#include <fst/shortest-distance.h>
#include <fst/shortest-path.h>
#include <fst/state-map.h>
#include <fst/statesort.h>
#include <fst/synchronize.h>
#include <fst/topsort.h>
#include <fst/union.h>
#include <fst/verify.h>
#include <fst/visit.h>

// Weights.
#include <fst/expectation-weight.h>
#include <fst/float-weight.h>
#include <fst/lexicographic-weight.h>
#include <fst/pair-weight.h>
#include <fst/power-weight.h>
#include <fst/product-weight.h>
#include <fst/signed-log-weight.h>
#include <fst/sparse-power-weight.h>
#include <fst/sparse-tuple-weight.h>
#include <fst/string-weight.h>
#include <fst/tuple-weight.h>
#include <fst/weight.h>

// Auxiliary classes for composition.
#include <fst/compose-filter.h>
#include <fst/lookahead-filter.h>
#include <fst/lookahead-matcher.h>
#include <fst/matcher-fst.h>
#include <fst/matcher.h>
#include <fst/state-table.h>

// Data structures.
#include <fst/heap.h>
#include <fst/interval-set.h>
#include <fst/queue.h>
#include <fst/union-find.h>

// Miscellaneous.
#include <fst/accumulator.h>
#include <fst/add-on.h>
#include <fst/arc.h>
#include <fst/arcfilter.h>
#include <fst/cache.h>
#include <fst/complement.h>
#include <fst/dfs-visit.h>
#include <fst/generic-register.h>
#include <fst/label-reachable.h>
#include <fst/partition.h>
#include <fst/properties.h>
#include <fst/register.h>
#include <fst/state-reachable.h>
#include <fst/string.h>
#include <fst/symbol-table.h>
#include <fst/symbol-table-ops.h>
#include <fst/test-properties.h>
#include <fst/util.h>


#endif  // FST_FSTLIB_H_
