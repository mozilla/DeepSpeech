// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/compact-fst.h>

namespace fst {

static FstRegisterer<
    CompactWeightedStringFst<StdArc, uint16>>
    CompactWeightedStringFst_StdArc_uint16_registerer;

static FstRegisterer<
    CompactWeightedStringFst<LogArc, uint16>>
    CompactWeightedStringFst_LogArc_uint16_registerer;

}  // namespace fst
