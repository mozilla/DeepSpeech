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
// Google-style logging declarations and inline definitions.

#ifndef FST_LIB_LOG_H_
#define FST_LIB_LOG_H_

#include <cassert>
#include <iostream>
#include <string>

#include <fst/types.h>
#include <fst/flags.h>

using std::string;

DECLARE_int32_t(v);

class LogMessage {
 public:
  LogMessage(const string &type) : fatal_(type == "FATAL") {
    std::cerr << type << ": ";
  }
  ~LogMessage() {
    std::cerr << std::endl;
    if(fatal_)
      exit(1);
  }
  std::ostream &stream() { return std::cerr; }

 private:
  bool fatal_;
};

#define LOG(type) LogMessage(#type).stream()
#define VLOG(level) if ((level) <= FLAGS_v) LOG(INFO)

// Checks
inline void FstCheck(bool x, const char* expr,
                const char *file, int line) {
  if (!x) {
    LOG(FATAL) << "Check failed: \"" << expr
               << "\" file: " << file
               << " line: " << line;
  }
}

#define CHECK(x) FstCheck(static_cast<bool>(x), #x, __FILE__, __LINE__)
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_NE(x, y) CHECK((x) != (y))

// Debug checks
#define DCHECK(x) assert(x)
#define DCHECK_EQ(x, y) DCHECK((x) == (y))
#define DCHECK_LT(x, y) DCHECK((x) < (y))
#define DCHECK_GT(x, y) DCHECK((x) > (y))
#define DCHECK_LE(x, y) DCHECK((x) <= (y))
#define DCHECK_GE(x, y) DCHECK((x) >= (y))
#define DCHECK_NE(x, y) DCHECK((x) != (y))


// Ports
#define ATTRIBUTE_DEPRECATED __attribute__((deprecated))

#endif  // FST_LIB_LOG_H_
