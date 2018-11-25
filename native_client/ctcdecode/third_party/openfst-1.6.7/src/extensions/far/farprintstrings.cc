// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Outputs as strings the string FSTs in a finite-state archive.

#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/extensions/far/farscript.h>
#include <fst/extensions/far/getters.h>

DEFINE_string(filename_prefix, "", "Prefix to append to filenames");
DEFINE_string(filename_suffix, "", "Suffix to append to filenames");
DEFINE_int32(generate_filenames, 0,
             "Generate N digit numeric filenames (def: use keys)");
DEFINE_string(begin_key, "",
              "First key to extract (def: first key in archive)");
DEFINE_string(end_key, "", "Last key to extract (def: last key in archive)");
// PrintStringsMain specific flag definitions.
DEFINE_bool(print_key, false, "Prefix each string by its key");
DEFINE_bool(print_weight, false, "Suffix each string by its weight");
DEFINE_string(entry_type, "line",
              "Entry type: one of : "
              "\"file\" (one FST per file), \"line\" (one FST per line)");
DEFINE_string(token_type, "symbol",
              "Token type: one of : "
              "\"symbol\", \"byte\", \"utf8\"");
DEFINE_string(symbols, "", "Label symbol table");
DEFINE_bool(initial_symbols, true,
            "Uses symbol table from the first Fst in archive for all entries.");

int main(int argc, char **argv) {
  namespace s = fst::script;

  string usage = "Print as string the string FSTs in an archive.\n\n  Usage:";
  usage += argv[0];
  usage += " [in1.far in2.far ...]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  s::ExpandArgs(argc, argv, &argc, &argv);

  std::vector<string> in_fnames;
  for (int i = 1; i < argc; ++i) in_fnames.push_back(argv[i]);
  if (in_fnames.empty()) in_fnames.push_back("");

  const auto arc_type = s::LoadArcTypeFromFar(in_fnames[0]);
  if (arc_type.empty()) return 1;

  fst::FarEntryType entry_type;
  if (!s::GetFarEntryType(FLAGS_entry_type, &entry_type)) {
    LOG(ERROR) << "Unknown or unsupported FAR entry type: " << FLAGS_entry_type;
    return 1;
  }

  fst::FarTokenType token_type;
  if (!s::GetFarTokenType(FLAGS_token_type, &token_type)) {
    LOG(ERROR) << "Unknown or unsupported FAR token type: " << FLAGS_token_type;
    return 1;
  }

  s::FarPrintStrings(in_fnames, arc_type, entry_type, token_type,
                     FLAGS_begin_key, FLAGS_end_key, FLAGS_print_key,
                     FLAGS_print_weight, FLAGS_symbols, FLAGS_initial_symbols,
                     FLAGS_generate_filenames, FLAGS_filename_prefix,
                     FLAGS_filename_suffix);

  return 0;
}
