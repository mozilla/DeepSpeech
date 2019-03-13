// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Minimizes a deterministic FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/minimize.h>

DECLARE_double(delta);
DECLARE_bool(allow_nondet);

int fstminimize_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::MutableFstClass;
  using fst::script::VectorFstClass;

  string usage = "Minimizes a deterministic FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out1.fst [out2.fst]]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 4) {
    ShowUsage();
    return 1;
  }

  const string in_name = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : "";
  const string out1_name =
      (argc > 2 && strcmp(argv[2], "-") != 0) ? argv[2] : "";
  const string out2_name =
      (argc > 3 && strcmp(argv[3], "-") != 0) ? argv[3] : "";

  if (out1_name.empty() && out2_name.empty() && argc > 3) {
    LOG(ERROR) << argv[0] << ": Both outputs can't be standard output.";
    return 1;
  }

  std::unique_ptr<MutableFstClass> fst1(MutableFstClass::Read(in_name, true));
  if (!fst1) return 1;

  if (argc > 3) {
    std::unique_ptr<MutableFstClass> fst2(new VectorFstClass(fst1->ArcType()));
    s::Minimize(fst1.get(), fst2.get(), FLAGS_delta, FLAGS_allow_nondet);
    if (!fst2->Write(out2_name)) return 1;
  } else {
    s::Minimize(fst1.get(), nullptr, FLAGS_delta, FLAGS_allow_nondet);
  }

  return !fst1->Write(out1_name);
}
