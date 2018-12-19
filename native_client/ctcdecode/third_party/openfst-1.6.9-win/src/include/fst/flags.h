// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
// 
// Google-style flag handling declarations and inline definitions.

#ifndef FST_LIB_FLAGS_H_
#define FST_LIB_FLAGS_H_

#include <cstdlib>

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <fst/types.h>
#include <fst/lock.h>

using std::string;

// FLAGS USAGE:
//
// Definition example:
//
//    DEFINE_int32_t(length, 0, "length");
//
// This defines variable FLAGS_length, initialized to 0.
//
// Declaration example:
//
//    DECLARE_int32_t(length);
//
// SET_FLAGS() can be used to set flags from the command line
// using, for example, '--length=2'.
//
// ShowUsage() can be used to print out command and flag usage.

#define DECLARE_bool(name) extern bool FLAGS_ ## name
#define DECLARE_string(name) extern string FLAGS_ ## name
#define DECLARE_int32_t(name) extern int32_t FLAGS_ ## name
#define DECLARE_int64_t(name) extern int64_t FLAGS_ ## name
#define DECLARE_double(name) extern double FLAGS_ ## name

template <typename T>
struct FlagDescription {
  FlagDescription(T *addr, const char *doc, const char *type,
		  const char *file, const T val)
      : address(addr),
    doc_string(doc),
    type_name(type),
    file_name(file),
    default_value(val) {}

  T *address;
  const char *doc_string;
  const char *type_name;
  const char *file_name;
  const T default_value;
};

template <typename T>
class FlagRegister {
 public:
  static FlagRegister<T> *GetRegister() {
    static auto reg = new FlagRegister<T>;
    return reg;
  }

  const FlagDescription<T> &GetFlagDescription(const string &name) const {
    fst::MutexLock l(&flag_lock_);
    auto it = flag_table_.find(name);
    return it != flag_table_.end() ? it->second : 0;
  }

  void SetDescription(const string &name,
                      const FlagDescription<T> &desc) {
    fst::MutexLock l(&flag_lock_);
    flag_table_.insert(make_pair(name, desc));
  }

  bool SetFlag(const string &val, bool *address) const {
    if (val == "true" || val == "1" || val.empty()) {
      *address = true;
      return true;
    } else if (val == "false" || val == "0") {
      *address = false;
      return true;
    }
    else {
      return false;
    }
  }

  bool SetFlag(const string &val, string *address) const {
    *address = val;
    return true;
  }

  bool SetFlag(const string &val, int32_t *address) const {
    char *p = 0;
    *address = strtol(val.c_str(), &p, 0);
    return !val.empty() && *p == '\0';
  }

  bool SetFlag(const string &val, int64_t *address) const {
    char *p = 0;
    *address = strtoll(val.c_str(), &p, 0);
    return !val.empty() && *p == '\0';
  }

  bool SetFlag(const string &val, double *address) const {
    char *p = 0;
    *address = strtod(val.c_str(), &p);
    return !val.empty() && *p == '\0';
  }

  bool SetFlag(const string &arg, const string &val) const {
    for (typename std::map< string, FlagDescription<T> >::const_iterator it =
           flag_table_.begin();
         it != flag_table_.end();
         ++it) {
      const string &name = it->first;
      const FlagDescription<T> &desc = it->second;
      if (arg == name)
        return SetFlag(val, desc.address);
    }
    return false;
  }

  void GetUsage(std::set<std::pair<string, string>> *usage_set) const {
    for (auto it = flag_table_.begin(); it != flag_table_.end(); ++it) {
      const string &name = it->first;
      const FlagDescription<T> &desc = it->second;
      string usage = "  --" + name;
      usage += ": type = ";
      usage += desc.type_name;
      usage += ", default = ";
      usage += GetDefault(desc.default_value) + "\n  ";
      usage += desc.doc_string;
      usage_set->insert(make_pair(desc.file_name, usage));
    }
  }

 private:
  string GetDefault(bool default_value) const {
    return default_value ? "true" : "false";
  }

  string GetDefault(const string &default_value) const {
    return "\"" + default_value + "\"";
  }

  template <class V>
  string GetDefault(const V &default_value) const {
    std::ostringstream strm;
    strm << default_value;
    return strm.str();
  }

  mutable fst::Mutex flag_lock_;        // Multithreading lock.
  std::map<string, FlagDescription<T>> flag_table_;
};

template <typename T>
class FlagRegisterer {
 public:
  FlagRegisterer(const string &name, const FlagDescription<T> &desc) {
    auto registr = FlagRegister<T>::GetRegister();
    registr->SetDescription(name, desc);
  }

 private:
  FlagRegisterer(const FlagRegisterer &) = delete;
  FlagRegisterer &operator=(const FlagRegisterer &) = delete;
};


#define DEFINE_VAR(type, name, value, doc)                                \
  type FLAGS_ ## name = value;                                            \
  static FlagRegisterer<type>                                             \
  name ## _flags_registerer(#name, FlagDescription<type>(&FLAGS_ ## name, \
                                                         doc,             \
                                                         #type,           \
                                                         __FILE__,        \
                                                         value))

#define DEFINE_bool(name, value, doc) DEFINE_VAR(bool, name, value, doc)
#define DEFINE_string(name, value, doc) \
  DEFINE_VAR(string, name, value, doc)
#define DEFINE_int32_t(name, value, doc) DEFINE_VAR(int32_t, name, value, doc)
#define DEFINE_int64_t(name, value, doc) DEFINE_VAR(int64_t, name, value, doc)
#define DEFINE_double(name, value, doc) DEFINE_VAR(double, name, value, doc)


// Temporary directory.
DECLARE_string(tmpdir);

void SetFlags(const char *usage, int *argc, char ***argv, bool remove_flags,
              const char *src = "");

#define SET_FLAGS(usage, argc, argv, rmflags) \
SetFlags(usage, argc, argv, rmflags, __FILE__)

// Deprecated; for backward compatibility.
inline void InitFst(const char *usage, int *argc, char ***argv, bool rmflags) {
  return SetFlags(usage, argc, argv, rmflags);
}

void ShowUsage(bool long_usage = true);

#endif  // FST_LIB_FLAGS_H_
