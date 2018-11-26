// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Creates a finite-state archive from input FSTs.

#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/extensions/far/farscript.h>
#include <fst/extensions/far/getters.h>
#include <fstream>

DEFINE_string(key_prefix, "", "Prefix to append to keys");
DEFINE_string(key_suffix, "", "Suffix to append to keys");
DEFINE_int32(generate_keys, 0,
             "Generate N digit numeric keys (def: use file basenames)");
DEFINE_string(far_type, "default",
              "FAR file format type: one of: \"default\", "
              "\"stlist\", \"sttable\"");
DEFINE_bool(file_list_input, false,
            "Each input file contains a list of files to be processed");

int main(int argc, char **argv) {
  namespace s = fst::script;

  string usage = "Creates a finite-state archive from input FSTs.\n\n Usage:";
  usage += argv[0];
  usage += " [in1.fst [[in2.fst ...] out.far]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  s::ExpandArgs(argc, argv, &argc, &argv);

  std::vector<string> in_fnames;
  if (FLAGS_file_list_input) {
    for (int i = 1; i < argc - 1; ++i) {
      std::ifstream istrm(argv[i]);
      string str;
      while (getline(istrm, str)) in_fnames.push_back(str);
    }
  } else {
    for (int i = 1; i < argc - 1; ++i)
      in_fnames.push_back(argv[i]);
  }
  if (in_fnames.empty())
    in_fnames.push_back(argc == 2 && strcmp(argv[1], "-") != 0 ? argv[1] : "");

  string out_fname =
      argc > 2 && strcmp(argv[argc - 1], "-") != 0 ? argv[argc - 1] : "";

  const auto arc_type = s::LoadArcTypeFromFst(in_fnames[0]);
  if (arc_type.empty()) return 1;

  const auto far_type = s::GetFarType(FLAGS_far_type);

  s::FarCreate(in_fnames, out_fname, arc_type, FLAGS_generate_keys, far_type,
               FLAGS_key_prefix, FLAGS_key_suffix);

  return 0;
}
