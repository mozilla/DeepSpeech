// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Two DFAs are equivalent iff their exit status is zero.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/equivalent.h>
#include <fst/script/getters.h>
#include <fst/script/randequivalent.h>

DECLARE_double(delta);
DECLARE_bool(random);
DECLARE_int32(max_length);
DECLARE_int32(npath);
DECLARE_int32(seed);
DECLARE_string(select);

int fstequivalent_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::RandGenOptions;
  using fst::script::FstClass;

  string usage =
      "Two DFAs are equivalent iff the exit status is zero.\n\n"
      "  Usage: ";
  usage += argv[0];
  usage += " in1.fst in2.fst\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc != 3) {
    ShowUsage();
    return 1;
  }

  const string in1_name = strcmp(argv[1], "-") == 0 ? "" : argv[1];
  const string in2_name = strcmp(argv[2], "-") == 0 ? "" : argv[2];

  if (in1_name.empty() && in2_name.empty()) {
    LOG(ERROR) << argv[0] << ": Can't take both inputs from standard input";
    return 1;
  }

  std::unique_ptr<FstClass> ifst1(FstClass::Read(in1_name));
  if (!ifst1) return 1;

  std::unique_ptr<FstClass> ifst2(FstClass::Read(in2_name));
  if (!ifst2) return 1;

  if (!FLAGS_random) {
    bool result = s::Equivalent(*ifst1, *ifst2, FLAGS_delta);
    if (!result) VLOG(1) << "FSTs are not equivalent";
    return result ? 0 : 2;
  } else {
    s::RandArcSelection ras;
    if (!s::GetRandArcSelection(FLAGS_select, &ras)) {
      LOG(ERROR) << argv[0] << ": Unknown or unsupported select type "
                            << FLAGS_select;
      return 1;
    }
    const RandGenOptions<s::RandArcSelection> opts(ras, FLAGS_max_length);
    bool result = s::RandEquivalent(*ifst1, *ifst2, FLAGS_npath, FLAGS_delta,
                                    FLAGS_seed, opts);
    if (!result) VLOG(1) << "FSTs are not equivalent";
    return result ? 0 : 2;
  }
}
