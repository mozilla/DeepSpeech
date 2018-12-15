// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Creates the union of two FSTs.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/script/union.h>

int fstunion_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::MutableFstClass;

  string usage = "Creates the union of two FSTs.\n\n  Usage: ";
  usage += argv[0];
  usage += " in1.fst in2.fst [out.fst]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc < 3 || argc > 4) {
    ShowUsage();
    return 1;
  }

  const string in1_name = strcmp(argv[1], "-") != 0 ? argv[1] : "";
  const string in2_name = strcmp(argv[2], "-") != 0 ? argv[2] : "";
  const string out_name = argc > 3 ? argv[3] : "";

  if (in1_name == "" && in2_name == "") {
    LOG(ERROR) << argv[0] << ": Can't take both inputs from standard input";
    return 1;
  }

  std::unique_ptr<MutableFstClass> fst1(MutableFstClass::Read(in1_name, true));
  if (!fst1) return 1;

  std::unique_ptr<FstClass> fst2(FstClass::Read(in2_name));
  if (!fst2) return 1;

  s::Union(fst1.get(), *fst2);

  return !fst1->Write(out_name);
}
