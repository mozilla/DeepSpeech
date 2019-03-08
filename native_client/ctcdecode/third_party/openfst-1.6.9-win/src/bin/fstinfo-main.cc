// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Prints out various information about an FST such as number of states
// and arcs and property values (see properties.h).

#include <cstring>

#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/script/info.h>

DECLARE_string(arc_filter);
DECLARE_string(info_type);
DECLARE_bool(pipe);
DECLARE_bool(test_properties);
DECLARE_bool(fst_verify);

int fstinfo_main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;

  string usage = "Prints out information about an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 2) {
    ShowUsage();
    return 1;
  }

  const string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  s::PrintFstInfo(*ifst, FLAGS_test_properties, FLAGS_arc_filter,
                  FLAGS_info_type, FLAGS_fst_verify, FLAGS_pipe);

  return 0;
}
