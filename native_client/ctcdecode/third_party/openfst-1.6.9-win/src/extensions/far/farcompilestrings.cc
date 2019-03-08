// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Compiles a set of stings as FSTs and stores them in a finite-state archive.

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
              "FAR file format type: one of: \"default\", \"fst\", "
              "\"stlist\", \"sttable\"");
DEFINE_bool(allow_negative_labels, false,
            "Allow negative labels (not recommended; may cause conflicts)");
DEFINE_string(arc_type, "standard", "Output arc type");
DEFINE_string(entry_type, "line",
              "Entry type: one of : "
              "\"file\" (one FST per file), \"line\" (one FST per line)");
DEFINE_string(fst_type, "vector", "Output FST type");
DEFINE_string(token_type, "symbol",
              "Token type: one of : "
              "\"symbol\", \"byte\", \"utf8\"");
DEFINE_string(symbols, "", "Label symbol table");
DEFINE_string(unknown_symbol, "", "");
DEFINE_bool(file_list_input, false,
            "Each input file contains a list of files to be processed");
DEFINE_bool(keep_symbols, false, "Store symbol table in the FAR file");
DEFINE_bool(initial_symbols, true,
            "When keep_symbols is true, stores symbol table only for the first"
            " FST in archive.");

int main(int argc, char **argv) {
  namespace s = fst::script;

  string usage = "Compiles a set of strings as FSTs and stores them in";
  usage += " a finite-state archive.\n\n  Usage:";
  usage += argv[0];
  usage += " [in1.txt [[in2.txt ...] out.far]]\n";

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
  if (in_fnames.empty()) {
    in_fnames.push_back(argc == 2 && strcmp(argv[1], "-") != 0 ? argv[1] : "");
  }

  string out_fname =
      argc > 2 && strcmp(argv[argc - 1], "-") != 0 ? argv[argc - 1] : "";

  fst::FarEntryType entry_type;
  if (!s::GetFarEntryType(FLAGS_entry_type, &entry_type)) {
    LOG(ERROR) << "Unknown or unsupported FAR entry type: " << FLAGS_entry_type;
    return 1;
  }

  fst::FarTokenType token_type;
  if (!s::GetFarTokenType(FLAGS_token_type, &token_type)) {
    LOG(ERROR) << "Unkonwn or unsupported FAR token type: " << FLAGS_token_type;
    return 1;
  }

  const auto far_type = s::GetFarType(FLAGS_far_type);

  s::FarCompileStrings(in_fnames, out_fname, FLAGS_arc_type, FLAGS_fst_type,
                       far_type, FLAGS_generate_keys, entry_type, token_type,
                       FLAGS_symbols, FLAGS_unknown_symbol, FLAGS_keep_symbols,
                       FLAGS_initial_symbols, FLAGS_allow_negative_labels,
                       FLAGS_key_prefix, FLAGS_key_suffix);

  return 0;
}
