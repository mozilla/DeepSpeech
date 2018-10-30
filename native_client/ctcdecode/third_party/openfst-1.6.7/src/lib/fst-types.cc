// Registration of common Fst and arc types.

#include <fst/arc.h>
#include <fst/compact-fst.h>
#include <fst/const-fst.h>
#include <fst/edit-fst.h>
#include <fst/register.h>
#include <fst/vector-fst.h>

namespace fst {

// Registers VectorFst, ConstFst and EditFst for common arcs types.
REGISTER_FST(VectorFst, StdArc);
REGISTER_FST(VectorFst, LogArc);
REGISTER_FST(VectorFst, Log64Arc);
REGISTER_FST(ConstFst, StdArc);
REGISTER_FST(ConstFst, LogArc);
REGISTER_FST(ConstFst, Log64Arc);
REGISTER_FST(EditFst, StdArc);
REGISTER_FST(EditFst, LogArc);
REGISTER_FST(EditFst, Log64Arc);

// Register CompactFst for common arcs with the default (uint32) size type
REGISTER_FST(CompactStringFst, StdArc);
REGISTER_FST(CompactStringFst, LogArc);
REGISTER_FST(CompactWeightedStringFst, StdArc);
REGISTER_FST(CompactWeightedStringFst, LogArc);
REGISTER_FST(CompactAcceptorFst, StdArc);
REGISTER_FST(CompactAcceptorFst, LogArc);
REGISTER_FST(CompactUnweightedFst, StdArc);
REGISTER_FST(CompactUnweightedFst, LogArc);
REGISTER_FST(CompactUnweightedAcceptorFst, StdArc);
REGISTER_FST(CompactUnweightedAcceptorFst, LogArc);

}  // namespace fst
