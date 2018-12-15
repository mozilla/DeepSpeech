// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Extracts component FSTs from an finite-state archive.

#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/extensions/far/farscript.h>
#include <fst/extensions/far/getters.h>

DEFINE_string(filename_prefix, "", "Prefix to append to filenames");
DEFINE_string(filename_suffix, "", "Suffix to append to filenames");
DEFINE_int32(generate_filenames, 0,
             "Generate N digit numeric filenames (def: use keys)");
DEFINE_string(keys, "",
              "Extract set of keys separated by comma (default) "
              "including ranges delimited by dash (default)");
DEFINE_string(key_separator, ",", "Separator for individual keys");
DEFINE_string(range_delimiter, "-", "Delimiter for ranges of keys");

int main(int argc, char **argv) {
  namespace s = fst::script;

  string usage = "Extracts FSTs from a finite-state archive.\n\n Usage:";
  usage += argv[0];
  usage += " [in1.far in2.far...]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  s::ExpandArgs(argc, argv, &argc, &argv);

  std::vector<string> in_fnames;
  for (int i = 1; i < argc; ++i) in_fnames.push_back(argv[i]);
  if (in_fnames.empty()) in_fnames.push_back("");

  const auto arc_type = s::LoadArcTypeFromFar(in_fnames[0]);
  if (arc_type.empty()) return 1;

  s::FarExtract(in_fnames, arc_type, FLAGS_generate_filenames, FLAGS_keys,
                FLAGS_key_separator, FLAGS_range_delimiter,
                FLAGS_filename_prefix, FLAGS_filename_suffix);

  return 0;
}
