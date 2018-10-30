// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Creates a finite-state archive from component FSTs.

#ifndef FST_EXTENSIONS_FAR_CREATE_H_
#define FST_EXTENSIONS_FAR_CREATE_H_

#include <libgen.h>
#include <sstream>
#include <string>
#include <vector>

#include <fst/extensions/far/far.h>

namespace fst {

template <class Arc>
void FarCreate(const std::vector<string> &in_fnames, const string &out_fname,
               const int32 generate_keys, const FarType &far_type,
               const string &key_prefix, const string &key_suffix) {
  std::unique_ptr<FarWriter<Arc>> far_writer(
      FarWriter<Arc>::Create(out_fname, far_type));
  if (!far_writer) return;
  for (size_t i = 0; i < in_fnames.size(); ++i) {
    std::unique_ptr<Fst<Arc>> ifst(Fst<Arc>::Read(in_fnames[i]));
    if (!ifst) return;
    string key;
    if (generate_keys > 0) {
      std::ostringstream keybuf;
      keybuf.width(generate_keys);
      keybuf.fill('0');
      keybuf << i + 1;
      key = keybuf.str();
    } else {
      auto *filename = new char[in_fnames[i].size() + 1];
      strcpy(filename, in_fnames[i].c_str());
      key = basename(filename);
      delete[] filename;
    }
    far_writer->Add(key_prefix + key + key_suffix, *ifst);
  }
}

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_CREATE_H_
