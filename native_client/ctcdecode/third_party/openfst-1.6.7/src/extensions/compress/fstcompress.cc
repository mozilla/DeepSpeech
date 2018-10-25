// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Compresses/decompresses an FST.

#include <memory>
#include <string>

#include <fst/extensions/compress/compress-script.h>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/util.h>
#include <fst/script/fst-class.h>

DEFINE_string(arc_type, "standard", "Output arc type");
DEFINE_bool(decode, false, "Decode");
DEFINE_bool(gzip, false,
            "Applies gzip compression after LZA compression and "
            "gzip decompression before LZA decompression "
            "(recommended)"
            "");

int main(int argc, char **argv) {
  namespace s = fst::script;

  using s::FstClass;
  using s::VectorFstClass;

  string usage = "Compresses/decompresses an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fstz]]\n";
  usage += " --decode [in.fstz [out.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  string in_name = (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  string out_name = argc > 2 ? argv[2] : "";

  if (FLAGS_decode == false) {
    std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
    if (!ifst) return 1;
    s::Compress(*ifst, out_name, FLAGS_gzip);
  } else {
    VectorFstClass ofst(FLAGS_arc_type);
    s::Decompress(in_name, &ofst, FLAGS_gzip);
    ofst.Write(out_name);
  }
  return 0;
}
