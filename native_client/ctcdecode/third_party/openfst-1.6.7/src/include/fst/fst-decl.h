// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// This file contains declarations of classes in the Fst template library.

#ifndef FST_FST_DECL_H_
#define FST_FST_DECL_H_

#include <sys/types.h>
#include <memory>  // for allocator<>

#include <fst/types.h>

namespace fst {

// Symbol table and iterator.

class SymbolTable;

class SymbolTableIterator;

// Weight templates and weights.

template <class T>
class FloatWeightTpl;

template <class T>
class TropicalWeightTpl;

template <class T>
class LogWeightTpl;

template <class T>
class MinMaxWeightTpl;

using FloatWeight = FloatWeightTpl<float>;

using TropicalWeight = TropicalWeightTpl<float>;

using LogWeight = LogWeightTpl<float>;

using MinMaxWeight = MinMaxWeightTpl<float>;

// Arc templates and arcs.

template <class Weight>
struct ArcTpl;

using StdArc = ArcTpl<TropicalWeight>;

using LogArc = ArcTpl<LogWeight>;

// Stores.

template <class Element, class U>
class DefaultCompactStore;

template <class Arc>
class DefaultCacheStore;

// FST templates.

template <class Arc, class Compactor, class U = uint32,
    class CompactStore = DefaultCompactStore<typename Compactor::Element, U>,
    class CacheStore = DefaultCacheStore<Arc>>
class CompactFst;

template <class Arc, class U = uint32>
class ConstFst;

template <class Arc, class Weight, class Matcher>
class EditFst;

template <class Arc>
class ExpandedFst;

template <class Arc>
class Fst;

template <class Arc>
class MutableFst;

template <class Arc, class Allocator = std::allocator<Arc>>
class VectorState;

template <class Arc, class State = VectorState<Arc>>
class VectorFst;

template <class Arc, class U = ssize_t>
class DefaultReplaceStateTable;

// On-the-fly operations.

template <class Arc, class Compare>
class ArcSortFst;

template <class Arc>
class ClosureFst;

template <class Arc, class Store = DefaultCacheStore<Arc>>
class ComposeFst;

template <class Arc>
class ConcatFst;

template <class Arc>
class DeterminizeFst;

template <class Arc>
class DifferenceFst;

template <class Arc>
class IntersectFst;

template <class Arc>
class InvertFst;

template <class AArc, class BArc, class Mapper>
class ArcMapFst;

template <class Arc>
class ProjectFst;

template <class AArc, class BArc, class Selector>
class RandGenFst;

template <class Arc>
class RelabelFst;

template <class Arc, class StateTable = DefaultReplaceStateTable<Arc>,
          class Store = DefaultCacheStore<Arc>>
class ReplaceFst;

template <class Arc>
class RmEpsilonFst;

template <class Arc>
class UnionFst;

// Heap.

template <class T, class Compare>
class Heap;

// Compactors.

template <class Arc>
class AcceptorCompactor;

template <class Arc>
class StringCompactor;

template <class Arc>
class UnweightedAcceptorCompactor;

template <class Arc>
class UnweightedCompactor;

template <class Arc>
class WeightedStringCompactor;

// Compact FSTs.

template <class Arc, class U = uint32>
using CompactStringFst = CompactFst<Arc, StringCompactor<Arc>, U>;

template <class Arc, class U = uint32>
using CompactWeightedStringFst =
    CompactFst<Arc, WeightedStringCompactor<Arc>, U>;

template <class Arc, class U = uint32>
using CompactAcceptorFst = CompactFst<Arc, AcceptorCompactor<Arc>, U>;

template <class Arc, class U = uint32>
using CompactUnweightedFst = CompactFst<Arc, UnweightedCompactor<Arc>, U>;

template <class Arc, class U = uint32>
using CompactUnweightedAcceptorFst =
    CompactFst<Arc, UnweightedAcceptorCompactor<Arc>, U>;

// StdArc aliases for FSTs.

using StdConstFst = ConstFst<StdArc>;
using StdExpandedFst = ExpandedFst<StdArc>;
using StdFst = Fst<StdArc>;
using StdMutableFst = MutableFst<StdArc>;
using StdVectorFst = VectorFst<StdArc>;

// StdArc aliases for on-the-fly operations.

template <class Compare>
using StdArcSortFst = ArcSortFst<StdArc, Compare>;

using StdClosureFst = ClosureFst<StdArc>;

using StdComposeFst = ComposeFst<StdArc>;

using StdConcatFst = ConcatFst<StdArc>;

using StdDeterminizeFst = DeterminizeFst<StdArc>;

using StdDifferenceFst = DifferenceFst<StdArc>;

using StdIntersectFst = IntersectFst<StdArc>;

using StdInvertFst = InvertFst<StdArc>;

using StdProjectFst = ProjectFst<StdArc>;

using StdRelabelFst = RelabelFst<StdArc>;

using StdReplaceFst = ReplaceFst<StdArc>;

using StdRmEpsilonFst = RmEpsilonFst<StdArc>;

using StdUnionFst = UnionFst<StdArc>;

// Filter states.

template <class T>
class IntegerFilterState;

using CharFilterState = IntegerFilterState<signed char>;

using ShortFilterState = IntegerFilterState<short>;  // NOLINT

using IntFilterState = IntegerFilterState<int>;

// Matchers and filters.

template <class FST>
class Matcher;

template <class Matcher1, class Matcher2 = Matcher1>
class NullComposeFilter;

template <class Matcher1, class Matcher2 = Matcher1>
class TrivialComposeFilter;

template <class Matcher1, class Matcher2 = Matcher1>
class SequenceComposeFilter;

template <class Matcher1, class Matcher2 = Matcher1>
class AltSequenceComposeFilter;

template <class Matcher1, class Matcher2 = Matcher1>
class MatchComposeFilter;

}  // namespace fst

#endif  // FST_FST_DECL_H_
