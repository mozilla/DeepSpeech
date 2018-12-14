// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/special/rho-fst.h>

#include <fst/fst.h>

DEFINE_int64(rho_fst_rho_label, 0,
             "Label of transitions to be interpreted as rho ('rest') "
             "transitions");
DEFINE_string(rho_fst_rewrite_mode, "auto",
              "Rewrite both sides when matching? One of:"
              " \"auto\" (rewrite iff acceptor), \"always\", \"never\"");

namespace fst {

const char rho_fst_type[] = "rho";
const char input_rho_fst_type[] = "input_rho";
const char output_rho_fst_type[] = "output_rho";

static FstRegisterer<StdRhoFst> RhoFst_StdArc_registerer;
static FstRegisterer<LogRhoFst> RhoFst_LogArc_registerer;
static FstRegisterer<Log64RhoFst> RhoFst_Log64Arc_registerer;

static FstRegisterer<StdInputRhoFst> InputRhoFst_StdArc_registerer;
static FstRegisterer<LogInputRhoFst> InputRhoFst_LogArc_registerer;
static FstRegisterer<Log64InputRhoFst> InputRhoFst_Log64Arc_registerer;

static FstRegisterer<StdOutputRhoFst> OutputRhoFst_StdArc_registerer;
static FstRegisterer<LogOutputRhoFst> OutputRhoFst_LogArc_registerer;
static FstRegisterer<Log64OutputRhoFst> OutputRhoFst_Log64Arc_registerer;

}  // namespace fst
