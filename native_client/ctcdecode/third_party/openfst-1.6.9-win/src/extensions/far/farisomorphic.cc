// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Tests if two Far files contains isomorphic (key,fst) pairs.

#include <string>

#include <fst/flags.h>
#include <fst/extensions/far/farscript.h>
#include <fst/extensions/far/getters.h>

DEFINE_string(begin_key, "",
              "First key to extract (def: first key in archive)");
DEFINE_string(end_key, "", "Last key to extract (def: last key in archive)");
DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");

int main(int argc, char **argv) {
  namespace s = fst::script;

  string usage = "Compares the FSTs in two FST archives for isomorphism.";
  usage += "\n\n  Usage:";
  usage += argv[0];
  usage += " in1.far in2.far\n";
  usage += "  Flags: begin_key end_key";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  s::ExpandArgs(argc, argv, &argc, &argv);

  if (argc != 3) {
    ShowUsage();
    return 1;
  }

  const auto arc_type = s::LoadArcTypeFromFar(argv[1]);
  if (arc_type.empty()) return 1;

  bool result = s::FarIsomorphic(argv[1], argv[2], arc_type,
                                 FLAGS_delta, FLAGS_begin_key, FLAGS_end_key);

  if (!result) VLOG(1) << "FARs are not isomorphic.";

  return result ? 0 : 2;
}
