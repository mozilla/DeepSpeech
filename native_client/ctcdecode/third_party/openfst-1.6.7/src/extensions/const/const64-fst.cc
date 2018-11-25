// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/const-fst.h>

namespace fst {

static FstRegisterer<ConstFst<StdArc, uint64>>
    ConstFst_StdArc_uint64_registerer;
static FstRegisterer<ConstFst<LogArc, uint64>>
    ConstFst_LogArc_uint64_registerer;
static FstRegisterer<ConstFst<Log64Arc, uint64>>
    ConstFst_Log64Arc_uint64_registerer;

}  // namespace fst
