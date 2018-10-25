// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_FAR_EQUAL_H_
#define FST_EXTENSIONS_FAR_EQUAL_H_

#include <memory>
#include <string>

#include <fst/extensions/far/far.h>
#include <fst/equal.h>

namespace fst {

template <class Arc>
bool FarEqual(const string &filename1, const string &filename2,
              float delta = kDelta, const string &begin_key = string(),
              const string &end_key = string()) {
  std::unique_ptr<FarReader<Arc>> reader1(FarReader<Arc>::Open(filename1));
  if (!reader1) {
    LOG(ERROR) << "FarEqual: Could not open FAR file " << filename1;
    return false;
  }
  std::unique_ptr<FarReader<Arc>> reader2(FarReader<Arc>::Open(filename2));
  if (!reader2) {
    LOG(ERROR) << "FarEqual: Could not open FAR file " << filename2;
    return false;
  }
  if (!begin_key.empty()) {
    bool find_begin1 = reader1->Find(begin_key);
    bool find_begin2 = reader2->Find(begin_key);
    if (!find_begin1 || !find_begin2) {
      bool ret = !find_begin1 && !find_begin2;
      if (!ret) {
        LOG(ERROR) << "FarEqual: Key " << begin_key << " missing from "
                   << (find_begin1 ? "second" : "first") << " archive";
      }
      return ret;
    }
  }
  for (; !reader1->Done() && !reader2->Done();
       reader1->Next(), reader2->Next()) {
    const auto &key1 = reader1->GetKey();
    const auto &key2 = reader2->GetKey();
    if (!end_key.empty() && end_key < key1 && end_key < key2) {
      return true;
    }
    if (key1 != key2) {
      LOG(ERROR) << "FarEqual: Mismatched keys " << key1 << " and " << key2;
      return false;
    }
    if (!Equal(*(reader1->GetFst()), *(reader2->GetFst()), delta)) {
      LOG(ERROR) << "FarEqual: FSTs for key " << key1 << " are not equal";
      return false;
    }
  }
  if (!reader1->Done() || !reader2->Done()) {
    LOG(ERROR) << "FarEqual: Key "
               << (reader1->Done() ? reader2->GetKey() : reader1->GetKey())
               << " missing from " << (reader2->Done() ? "first" : "second")
               << " archive";
    return false;
  }
  return true;
}

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_EQUAL_H_
