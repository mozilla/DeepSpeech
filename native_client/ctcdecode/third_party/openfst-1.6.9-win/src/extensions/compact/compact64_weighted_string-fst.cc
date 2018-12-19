// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/compact-fst.h>

namespace fst {

static FstRegisterer<
    CompactWeightedStringFst<StdArc, uint64>>
    CompactWeightedStringFst_StdArc_uint64_registerer;
static FstRegisterer<
    CompactWeightedStringFst<LogArc, uint64>>
    CompactWeightedStringFst_LogArc_uint64_registerer;

}  // namespace fst
