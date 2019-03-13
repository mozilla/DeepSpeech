// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/linear/linear-fst.h>
#include <fst/register.h>

using fst::LinearClassifierFst;
using fst::StdArc;
using fst::LogArc;

REGISTER_FST(LinearClassifierFst, StdArc);
REGISTER_FST(LinearClassifierFst, LogArc);
