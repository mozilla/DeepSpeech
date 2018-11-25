// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/compact-fst.h>

namespace fst {

static FstRegisterer<CompactAcceptorFst<StdArc, uint16>>
    CompactAcceptorFst_StdArc_uint16_registerer;
static FstRegisterer<CompactAcceptorFst<LogArc, uint16>>
    CompactAcceptorFst_LogArc_uint16_registerer;

}  // namespace fst
