// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/fst.h>
#include <fst/extensions/special/phi-fst.h>

DEFINE_int64(phi_fst_phi_label, 0,
             "Label of transitions to be interpreted as phi ('failure') "
              "transitions");
DEFINE_bool(phi_fst_phi_loop, true,
            "When true, a phi self loop consumes a symbol");
DEFINE_string(phi_fst_rewrite_mode, "auto",
              "Rewrite both sides when matching? One of:"
              " \"auto\" (rewrite iff acceptor), \"always\", \"never\"");

namespace fst {

const char phi_fst_type[] = "phi";
const char input_phi_fst_type[] = "input_phi";
const char output_phi_fst_type[] = "output_phi";

static FstRegisterer<StdPhiFst> PhiFst_StdArc_registerer;
static FstRegisterer<LogPhiFst> PhiFst_LogArc_registerer;
static FstRegisterer<Log64PhiFst> PhiFst_Log64Arc_registerer;

static FstRegisterer<StdInputPhiFst> InputPhiFst_StdArc_registerer;
static FstRegisterer<LogInputPhiFst> InputPhiFst_LogArc_registerer;
static FstRegisterer<Log64InputPhiFst> InputPhiFst_Log64Arc_registerer;

static FstRegisterer<StdOutputPhiFst> OutputPhiFst_StdArc_registerer;
static FstRegisterer<LogOutputPhiFst> OutputPhiFst_LogArc_registerer;
static FstRegisterer<Log64OutputPhiFst> OutputPhiFst_Log64Arc_registerer;

}  // namespace fst
