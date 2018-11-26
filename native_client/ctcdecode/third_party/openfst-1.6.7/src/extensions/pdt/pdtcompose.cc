// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Composes a PDT and an FST.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/extensions/pdt/getters.h>
#include <fst/extensions/pdt/pdtscript.h>
#include <fst/util.h>

DEFINE_string(pdt_parentheses, "", "PDT parenthesis label pairs");
DEFINE_bool(left_pdt, true, "Is the first argument the PDT?");
DEFINE_bool(connect, true, "Trim output?");
DEFINE_string(compose_filter, "paren",
              "Composition filter, one of: \"expand\", \"expand_paren\", "
              "\"paren\"");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::ReadLabelPairs;
  using fst::PdtComposeFilter;
  using fst::PdtComposeOptions;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;

  string usage = "Compose a PDT and an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " in.pdt in.fst [out.pdt]\n";
  usage += " in.fst in.pdt [out.pdt]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc < 3 || argc > 4) {
    ShowUsage();
    return 1;
  }

  const string in1_name = strcmp(argv[1], "-") == 0 ? "" : argv[1];
  const string in2_name = strcmp(argv[2], "-") == 0 ? "" : argv[2];
  const string out_name = argc > 3 ? argv[3] : "";

  if (in1_name.empty() && in2_name.empty()) {
    LOG(ERROR) << argv[0] << ": Can't take both inputs from standard input.";
    return 1;
  }

  std::unique_ptr<FstClass> ifst1(FstClass::Read(in1_name));
  if (!ifst1) return 1;
  std::unique_ptr<FstClass> ifst2(FstClass::Read(in2_name));
  if (!ifst2) return 1;

  if (FLAGS_pdt_parentheses.empty()) {
    LOG(ERROR) << argv[0] << ": No PDT parenthesis label pairs provided";
    return 1;
  }

  std::vector<s::LabelPair> parens;
  if (!ReadLabelPairs(FLAGS_pdt_parentheses, &parens, false)) return 1;

  VectorFstClass ofst(ifst1->ArcType());

  PdtComposeFilter compose_filter;
  if (!s::GetPdtComposeFilter(FLAGS_compose_filter, &compose_filter)) {
    LOG(ERROR) << argv[0] << ": Unknown or unsupported compose filter type: "
               << FLAGS_compose_filter;
    return 1;
  }

  const PdtComposeOptions copts(FLAGS_connect, compose_filter);

  s::PdtCompose(*ifst1, *ifst2, parens, &ofst, copts, FLAGS_left_pdt);

  ofst.Write(out_name);

  return 0;
}
