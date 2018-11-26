// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Disambiguates an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/disambiguate.h>

DECLARE_double(delta);
DECLARE_int64(nstate);
DECLARE_string(weight);
DECLARE_int64(subsequential_label);

int fstdisambiguate_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;
  using fst::script::WeightClass;

  string usage = "Disambiguates an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  VectorFstClass ofst(ifst->ArcType());

  const auto weight_threshold =
      FLAGS_weight.empty() ? WeightClass::Zero(ifst->WeightType())
                           : WeightClass(ifst->WeightType(), FLAGS_weight);

  const s::DisambiguateOptions opts(FLAGS_delta, weight_threshold, FLAGS_nstate,
                                    FLAGS_subsequential_label);

  s::Disambiguate(*ifst, &ofst, opts);

  return !ofst.Write(out_name);
}
