// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Pushes weights and/or output labels in an FST toward the initial or final
// states.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/getters.h>
#include <fst/script/push.h>

DECLARE_double(delta);
DECLARE_bool(push_weights);
DECLARE_bool(push_labels);
DECLARE_bool(remove_total_weight);
DECLARE_bool(remove_common_affix);
DECLARE_bool(to_final);

int fstpush_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;

  string usage = "Pushes weights and/or olabels in an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  string in_name = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : "";
  string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  const auto flags =
      s::GetPushFlags(FLAGS_push_weights, FLAGS_push_labels,
                      FLAGS_remove_total_weight, FLAGS_remove_common_affix);

  VectorFstClass ofst(ifst->ArcType());

  s::Push(*ifst, &ofst, flags, s::GetReweightType(FLAGS_to_final),
          FLAGS_delta);

  return !ofst.Write(out_name);
}
