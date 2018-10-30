// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Returns the shortest path in a (bounded-stack) PDT.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/extensions/pdt/pdtscript.h>
#include <fst/util.h>

DEFINE_bool(keep_parentheses, false, "Keep PDT parentheses in result?");
DEFINE_string(queue_type, "fifo",
              "Queue type: one of: "
              "\"fifo\", \"lifo\", \"state\"");
DEFINE_bool(path_gc, true, "Garbage collect shortest path data?");
DEFINE_string(pdt_parentheses, "", "PDT parenthesis label pairs");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;
  using fst::QueueType;
  using fst::ReadLabelPairs;

  string usage = "Shortest path in a (bounded-stack) PDT.\n\n  Usage: ";
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

  if (FLAGS_pdt_parentheses.empty()) {
    LOG(ERROR) << argv[0] << ": No PDT parenthesis label pairs provided";
    return 1;
  }

  std::vector<s::LabelPair> parens;
  if (!ReadLabelPairs(FLAGS_pdt_parentheses, &parens, false)) return 1;

  VectorFstClass ofst(ifst->ArcType());

  QueueType qt;
  if (FLAGS_queue_type == "fifo") {
    qt = fst::FIFO_QUEUE;
  } else if (FLAGS_queue_type == "lifo") {
    qt = fst::LIFO_QUEUE;
  } else if (FLAGS_queue_type == "state") {
    qt = fst::STATE_ORDER_QUEUE;
  } else {
    LOG(ERROR) << "Unknown queue type: " << FLAGS_queue_type;
    return 1;
  }

  const s::PdtShortestPathOptions opts(qt, FLAGS_keep_parentheses,
                                       FLAGS_path_gc);

  s::PdtShortestPath(*ifst, parens, &ofst, opts);

  ofst.Write(out_name);

  return 0;
}
