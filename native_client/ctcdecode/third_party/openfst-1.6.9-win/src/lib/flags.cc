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
// Google-style flag handling definitions.

#include <cstring>

#include <fst/compat.h>
#include <fst/flags.h>

static const char *private_tmpdir = getenv("TMPDIR");

DEFINE_int32_t(v, 0, "verbosity level");
DEFINE_bool(help, false, "show usage information");
DEFINE_bool(helpshort, false, "show brief usage information");
#ifndef _MSC_VER
DEFINE_string(tmpdir, private_tmpdir ? private_tmpdir : "/tmp",
              "temporary directory");
#else
DEFINE_string(tmpdir, private_tmpdir ? private_tmpdir : getenv("TEMP"),
              "temporary directory");
#endif // !_MSC_VER

using namespace std;

static string flag_usage;
static string prog_src;

// Sets prog_src to src.
static void SetProgSrc(const char *src) {
  prog_src = src;
  // Remove "-main" in src filename. Flags are defined in fstx.cc but SetFlags()
  // is called in fstx-main.cc, which results in a filename mismatch in
  // ShowUsageRestrict() below.
  static constexpr char kMainSuffix[] = "-main.cc";
  const int prefix_length = prog_src.size() - strlen(kMainSuffix);
  if (prefix_length > 0 && prog_src.substr(prefix_length) == kMainSuffix) {
    prog_src.erase(prefix_length, strlen("-main"));
  }
}

void SetFlags(const char *usage, int *argc, char ***argv,
              bool remove_flags, const char *src) {
  flag_usage = usage;
  SetProgSrc(src);

  int index = 1;
  for (; index < *argc; ++index) {
    string argval = (*argv)[index];
    if (argval[0] != '-' || argval == "-") break;
    while (argval[0] == '-') argval = argval.substr(1);  // Removes initial '-'.
    string arg = argval;
    string val = "";
    // Splits argval (arg=val) into arg and val.
    auto pos = argval.find("=");
    if (pos != string::npos) {
      arg = argval.substr(0, pos);
      val = argval.substr(pos + 1);
    }
    auto bool_register = FlagRegister<bool>::GetRegister();
    if (bool_register->SetFlag(arg, val))
      continue;
    auto string_register = FlagRegister<string>::GetRegister();
    if (string_register->SetFlag(arg, val))
      continue;
    auto int32_t_register = FlagRegister<int32_t>::GetRegister();
    if (int32_t_register->SetFlag(arg, val))
      continue;
    auto int64_t_register = FlagRegister<int64_t>::GetRegister();
    if (int64_t_register->SetFlag(arg, val))
      continue;
    auto double_register = FlagRegister<double>::GetRegister();
    if (double_register->SetFlag(arg, val))
      continue;
    LOG(FATAL) << "SetFlags: Bad option: " << (*argv)[index];
  }
  if (remove_flags) {
    for (auto i = 0; i < *argc - index; ++i) {
      (*argv)[i + 1] = (*argv)[i + index];
    }
    *argc -= index - 1;
  }
  if (FLAGS_help) {
    ShowUsage(true);
    exit(1);
  }
  if (FLAGS_helpshort) {
    ShowUsage(false);
    exit(1);
  }
}

// If flag is defined in file 'src' and 'in_src' true or is not
// defined in file 'src' and 'in_src' is false, then print usage.
static void
ShowUsageRestrict(const std::set<pair<string, string>> &usage_set,
		  const string &src, bool in_src, bool show_file) {
  string old_file;
  bool file_out = false;
  bool usage_out = false;
  for (const auto &pair : usage_set) {
    const auto &file = pair.first;
    const auto &usage = pair.second;
    bool match = file == src;
    if ((match && !in_src) || (!match && in_src)) continue;
    if (file != old_file) {
      if (show_file) {
        if (file_out) cout << "\n";
	    cout << "Flags from: " << file << "\n";
        file_out = true;
      }
      old_file = file;
    }
    cout << usage << "\n";
    usage_out = true;
  }
  if (usage_out) cout << "\n";
}

void ShowUsage(bool long_usage) {
  std::set<pair<string, string>> usage_set;
  cout << flag_usage << "\n";
  auto bool_register = FlagRegister<bool>::GetRegister();
  bool_register->GetUsage(&usage_set);
  auto string_register = FlagRegister<string>::GetRegister();
  string_register->GetUsage(&usage_set);
  auto int32_t_register = FlagRegister<int32_t>::GetRegister();
  int32_t_register->GetUsage(&usage_set);
  auto int64_t_register = FlagRegister<int64_t>::GetRegister();
  int64_t_register->GetUsage(&usage_set);
  auto double_register = FlagRegister<double>::GetRegister();
  double_register->GetUsage(&usage_set);
  if (!prog_src.empty()) {
    cout << "PROGRAM FLAGS:\n\n";
    ShowUsageRestrict(usage_set, prog_src, true, false);
  }
  if (!long_usage) return;
  if (!prog_src.empty()) cout << "LIBRARY FLAGS:\n\n";
  ShowUsageRestrict(usage_set, prog_src, false, true);
}
