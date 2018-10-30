// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/special/sigma-fst.h>

#include <fst/fst.h>

DEFINE_int64(sigma_fst_sigma_label, 0,
             "Label of transitions to be interpreted as sigma ('any') "
             "transitions");
DEFINE_string(sigma_fst_rewrite_mode, "auto",
              "Rewrite both sides when matching? One of:"
              " \"auto\" (rewrite iff acceptor), \"always\", \"never\"");

namespace fst {

const char sigma_fst_type[] = "sigma";
const char input_sigma_fst_type[] = "input_sigma";
const char output_sigma_fst_type[] = "output_sigma";

static FstRegisterer<StdSigmaFst> SigmaFst_StdArc_registerer;
static FstRegisterer<LogSigmaFst> SigmaFst_LogArc_registerer;
static FstRegisterer<Log64SigmaFst> SigmaFst_Log64Arc_registerer;

static FstRegisterer<StdInputSigmaFst> InputSigmaFst_StdArc_registerer;
static FstRegisterer<LogInputSigmaFst> InputSigmaFst_LogArc_registerer;
static FstRegisterer<Log64InputSigmaFst> InputSigmaFst_Log64Arc_registerer;

static FstRegisterer<StdOutputSigmaFst> OutputSigmaFst_StdArc_registerer;
static FstRegisterer<LogOutputSigmaFst> OutputSigmaFst_LogArc_registerer;
static FstRegisterer<Log64OutputSigmaFst> OutputSigmaFst_Log64Arc_registerer;

}  // namespace fst
