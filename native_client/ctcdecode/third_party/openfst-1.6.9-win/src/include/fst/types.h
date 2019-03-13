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
// Various type definitions (mostly for Google compatibility).

#include <cstddef>       // For std::ptrdiff_t.
#include <cstdlib>       // for ssize_t.
#include <cstdint>       // for ?int*_t.

#ifndef FST_LIB_TYPES_H_
#define FST_LIB_TYPES_H_


//using ssize_t = std::ptrdiff_t;
//#ifdef _MSC_VER
// Not really Windows-specific: they should have used ptrdiff_t in the first
// place. But on Windows there has never been ssize_t.
//using ssize_t = std::ptrdiff_t;
//#endif  // _MSC_VER

#endif  // FST_LIB_TYPES_H_
