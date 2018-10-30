// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/const-fst.h>

namespace fst {

static FstRegisterer<ConstFst<StdArc, uint16>>
    ConstFst_StdArc_uint16_registerer;
static FstRegisterer<ConstFst<LogArc, uint16>>
    ConstFst_LogArc_uint16_registerer;
static FstRegisterer<ConstFst<Log64Arc, uint16>>
    ConstFst_Log64Arc_uint16_registerer;

}  // namespace fst
