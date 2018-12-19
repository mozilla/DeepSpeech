// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Subtracts an unweighted DFA from an FSA.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/difference.h>

DECLARE_string(compose_filter);
DECLARE_bool(connect);

int fstdifference_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::ComposeFilter;
  using fst::DifferenceOptions;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;

  string usage = "Subtracts an unweighted DFA from an FSA.\n\n  Usage: ";
  usage += argv[0];
  usage += " in1.fst in2.fst [out.fst]\n";

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
    LOG(ERROR) << argv[0] << ": Can't take both inputs from standard input";
    return 1;
  }

  std::unique_ptr<FstClass> ifst1(FstClass::Read(in1_name));
  if (!ifst1) return 1;

  std::unique_ptr<FstClass> ifst2(FstClass::Read(in2_name));
  if (!ifst2) return 1;

  VectorFstClass ofst(ifst1->ArcType());

  ComposeFilter compose_filter;
  if (!s::GetComposeFilter(FLAGS_compose_filter, &compose_filter)) {
    LOG(ERROR) << argv[0] << ": Unknown or unsupported compose filter type: "
               << FLAGS_compose_filter;
    return 1;
  }

  const DifferenceOptions opts(FLAGS_connect, compose_filter);

  s::Difference(*ifst1, *ifst2, &ofst, opts);

  return !ofst.Write(out_name);
}
