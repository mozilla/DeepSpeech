// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Reverses an MPDT.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/extensions/mpdt/mpdtscript.h>
#include <fst/extensions/mpdt/read_write_utils.h>
#include <fst/util.h>

DEFINE_string(mpdt_parentheses, "",
              "MPDT parenthesis label pairs with assignments.");

DEFINE_string(mpdt_new_parentheses, "",
              "Output for reassigned parentheses and stacks");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::ReadLabelTriples;
  using fst::WriteLabelTriples;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;

  string usage = "Reverse an MPDT.\n\n  Usage: ";
  usage += argv[0];
  usage += " in.pdt [out.fst]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  if (FLAGS_mpdt_parentheses.empty()) {
    LOG(ERROR) << argv[0] << ": No MPDT parenthesis label pairs provided";
    return 1;
  }

  if (FLAGS_mpdt_new_parentheses.empty()) {
    LOG(ERROR) << argv[0] << ": No MPDT output parenthesis label file provided";
    return 1;
  }

  std::vector<s::LabelPair> parens;
  std::vector<int64> assignments;
  if (!ReadLabelTriples(FLAGS_mpdt_parentheses, &parens, &assignments, false))
    return 1;

  VectorFstClass ofst(ifst->ArcType());

  s::MPdtReverse(*ifst, parens, &assignments, &ofst);

  ofst.Write(out_name);

  if (!WriteLabelTriples(FLAGS_mpdt_new_parentheses, parens, assignments))
    return 1;

  return 0;
}
