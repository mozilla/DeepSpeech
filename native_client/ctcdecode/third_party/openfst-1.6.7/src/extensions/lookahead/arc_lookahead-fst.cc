// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/matcher-fst.h>

namespace fst {

static FstRegisterer<StdArcLookAheadFst> ArcLookAheadFst_StdArc_registerer;
static FstRegisterer<MatcherFst<
    ConstFst<LogArc>, ArcLookAheadMatcher<SortedMatcher<ConstFst<LogArc>>>,
    arc_lookahead_fst_type>>
    ArcLookAheadFst_LogArc_registerer;
static FstRegisterer<MatcherFst<
    ConstFst<Log64Arc>, ArcLookAheadMatcher<SortedMatcher<ConstFst<Log64Arc>>>,
    arc_lookahead_fst_type>>
    ArcLookAheadFst_Log64Arc_registerer;

}  // namespace fst
