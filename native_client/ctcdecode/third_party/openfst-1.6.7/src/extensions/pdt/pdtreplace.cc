// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Converts an RTN represented by FSTs and non-terminal labels into a PDT.

#include <cstring>

#include <string>
#include <vector>

#include <fst/flags.h>

#include <fst/extensions/pdt/getters.h>
#include <fst/extensions/pdt/pdtscript.h>
#include <fst/util.h>
#include <fst/vector-fst.h>

DEFINE_string(pdt_parentheses, "", "PDT parenthesis label pairs");
DEFINE_string(pdt_parser_type, "left",
              "Construction method, one of: \"left\", \"left_sr\"");
DEFINE_int64(start_paren_labels, fst::kNoLabel,
             "Index to use for the first inserted parentheses; if not "
             "specified, the next available label beyond the highest output "
             "label is used");
DEFINE_string(left_paren_prefix, "(_", "Prefix to attach to SymbolTable "
              "labels for inserted left parentheses");
DEFINE_string(right_paren_prefix, ")_", "Prefix to attach to SymbolTable "
              "labels for inserted right parentheses");

void Cleanup(std::vector<fst::script::LabelFstClassPair> *pairs) {
  for (const auto &pair : *pairs) {
    delete pair.second;
  }
  pairs->clear();
}

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;
  using fst::PdtParserType;
  using fst::WriteLabelPairs;

  string usage = "Converts an RTN represented by FSTs";
  usage += " and non-terminal labels into PDT.\n\n  Usage: ";
  usage += argv[0];
  usage += " root.fst rootlabel [rule1.fst label1 ...] [out.fst]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc < 4) {
    ShowUsage();
    return 1;
  }

  const string in_name = argv[1];
  const string out_name = argc % 2 == 0 ? argv[argc - 1] : "";

  auto *ifst = FstClass::Read(in_name);
  if (!ifst) return 1;

  PdtParserType parser_type;
  if (!s::GetPdtParserType(FLAGS_pdt_parser_type, &parser_type)) {
    LOG(ERROR) << argv[0] << ": Unknown PDT parser type: "
               << FLAGS_pdt_parser_type;
    delete ifst;
    return 1;
  }

  std::vector<s::LabelFstClassPair> pairs;
  // Note that if the root label is beyond the range of the underlying FST's
  // labels, truncation will occur.
  const auto root = atoll(argv[2]);
  pairs.emplace_back(root, ifst);

  for (auto i = 3; i < argc - 1; i += 2) {
    ifst = FstClass::Read(argv[i]);
    if (!ifst) {
      Cleanup(&pairs);
      return 1;
    }
    // Note that if the root label is beyond the range of the underlying FST's
    // labels, truncation will occur.
    const auto label = atoll(argv[i + 1]);
    pairs.emplace_back(label, ifst);
  }

  VectorFstClass ofst(ifst->ArcType());
  std::vector<s::LabelPair> parens;
  s::PdtReplace(pairs, &ofst, &parens, root, parser_type,
                FLAGS_start_paren_labels, FLAGS_left_paren_prefix,
                FLAGS_right_paren_prefix);
  Cleanup(&pairs);

  if (!FLAGS_pdt_parentheses.empty()) {
    if (!WriteLabelPairs(FLAGS_pdt_parentheses, parens)) return 1;
  }

  ofst.Write(out_name);

  return 0;
}
