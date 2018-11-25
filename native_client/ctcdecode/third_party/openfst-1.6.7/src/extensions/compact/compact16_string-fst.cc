// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/compact-fst.h>

namespace fst {

static FstRegisterer<CompactStringFst<StdArc, uint16>>
    CompactStringFst_StdArc_uint16_registerer;
static FstRegisterer<CompactStringFst<LogArc, uint16>>
    CompactStringFst_LogArc_uint16_registerer;

}  // namespace fst
