// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/matcher-fst.h>

namespace fst {

static FstRegisterer<StdOLabelLookAheadFst>
    OLabelLookAheadFst_StdArc_registerer;
static FstRegisterer<MatcherFst<
    ConstFst<LogArc>,
    LabelLookAheadMatcher<SortedMatcher<ConstFst<LogArc>>,
                          olabel_lookahead_flags, FastLogAccumulator<LogArc>>,
    olabel_lookahead_fst_type, LabelLookAheadRelabeler<LogArc>>>
    OLabelLookAheadFst_LogArc_registerer;
static FstRegisterer<MatcherFst<
    ConstFst<Log64Arc>,
    LabelLookAheadMatcher<SortedMatcher<ConstFst<Log64Arc>>,
                          olabel_lookahead_flags, FastLogAccumulator<Log64Arc>>,
    olabel_lookahead_fst_type, LabelLookAheadRelabeler<Log64Arc>>>
    OLabelLookAheadFst_Log64Arc_registerer;

}  // namespace fst
