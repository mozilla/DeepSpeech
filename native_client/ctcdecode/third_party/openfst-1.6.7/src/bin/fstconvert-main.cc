// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Converts an FST to another type.

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/convert.h>

DECLARE_string(fst_type);

int fstconvert_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;

  string usage = "Converts an FST to another type.\n\n  Usage: ";
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

  if (ifst->FstType() != FLAGS_fst_type) {
    std::unique_ptr<FstClass> ofst(s::Convert(*ifst, FLAGS_fst_type));
    if (!ofst) return 1;
    return !ofst->Write(out_name);
  } else {
    return !ifst->Write(out_name);
  }
}
