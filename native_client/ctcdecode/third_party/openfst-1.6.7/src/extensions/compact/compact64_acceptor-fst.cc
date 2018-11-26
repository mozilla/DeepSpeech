// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/compact-fst.h>

namespace fst {

static FstRegisterer<CompactAcceptorFst<StdArc, uint64>>
    CompactAcceptorFst_StdArc_uint64_registerer;

static FstRegisterer<CompactAcceptorFst<LogArc, uint64>>
    CompactAcceptorFst_LogArc_uint64_registerer;

}  // namespace fst
