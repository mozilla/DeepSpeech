// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Definitions and functions for invoking and using Far main functions that
// support multiple and extensible arc types.

#include <fst/extensions/far/script-impl.h>

#include <string>

#include <fst/extensions/far/far.h>
#include <fstream>

namespace fst {
namespace script {

string LoadArcTypeFromFar(const string &far_fname) {
  FarHeader hdr;
  if (!hdr.Read(far_fname)) {
    LOG(ERROR) << "Error reading FAR: " << far_fname;
    return "";
  }
  string atype = hdr.ArcType();
  if (atype == "unknown") {
    LOG(ERROR) << "Empty FST archive: " << far_fname;
    return "";
  }
  return atype;
}

string LoadArcTypeFromFst(const string &fst_fname) {
  FstHeader hdr;
  std::ifstream in(fst_fname, std::ios_base::in | std::ios_base::binary);
  if (!hdr.Read(in, fst_fname)) {
    LOG(ERROR) << "Error reading FST: " << fst_fname;
    return "";
  }
  return hdr.ArcType();
}

}  // namespace script
}  // namespace fst
