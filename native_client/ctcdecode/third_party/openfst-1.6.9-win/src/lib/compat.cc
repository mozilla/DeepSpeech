// compat.cc
//
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
// Author: riley@google.com (Michael Riley)
//
// \file
// Google compatibility definitions.

#include <cstring>
#include <fst/compat.h>

using namespace std;

void FailedNewHandler() {
  cerr << "Memory allocation failed\n";
  exit(1);
}

#ifdef _MSC_VER

const char *basename(const char *path) {
  //the man page for the original basename states the function
  //can return a pointer to an internal static structure
  //so this might be ugly but still within the scope of acceptable behavior
  char basename[_MAX_FNAME];
  char ext[_MAX_EXT];
  static char full_path[_MAX_EXT + _MAX_FNAME];

  _splitpath(path, NULL, NULL, basename, ext);
  _makepath(full_path, NULL, NULL, basename, ext);
  return full_path;
}

#endif  // _MSC_VER
