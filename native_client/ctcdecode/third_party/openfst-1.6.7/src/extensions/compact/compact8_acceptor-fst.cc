// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/compact-fst.h>

namespace fst {

static FstRegisterer<CompactAcceptorFst<StdArc, uint8>>
    CompactAcceptorFst_StdArc_uint8_registerer;
static FstRegisterer<CompactAcceptorFst<LogArc, uint8>>
    CompactAcceptorFst_LogArc_uint8_registerer;

}  // namespace fst
