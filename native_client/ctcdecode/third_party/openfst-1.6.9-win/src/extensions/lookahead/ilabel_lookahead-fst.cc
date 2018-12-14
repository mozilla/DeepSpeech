// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/matcher-fst.h>

namespace fst {

static FstRegisterer<StdILabelLookAheadFst>
    ILabelLookAheadFst_StdArc_registerer;
static FstRegisterer<MatcherFst<
    ConstFst<LogArc>,
    LabelLookAheadMatcher<SortedMatcher<ConstFst<LogArc>>,
                          ilabel_lookahead_flags, FastLogAccumulator<LogArc>>,
    ilabel_lookahead_fst_type, LabelLookAheadRelabeler<LogArc>>>
    ILabelLookAheadFst_LogArc_registerer;
static FstRegisterer<MatcherFst<
    ConstFst<Log64Arc>,
    LabelLookAheadMatcher<SortedMatcher<ConstFst<Log64Arc>>,
                          ilabel_lookahead_flags, FastLogAccumulator<Log64Arc>>,
    ilabel_lookahead_fst_type, LabelLookAheadRelabeler<Log64Arc>>>
    ILabelLookAheadFst_Log64Arc_registerer;

}  // namespace fst
