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
// Google-compatibility locking declarations and inline definitions.

#ifndef FST_LIB_LOCK_H_
#define FST_LIB_LOCK_H_

#include <mutex>

namespace fst {

using namespace std;

class Mutex {
 public:
  Mutex() {}

  inline void Lock() { mu_.lock(); }

  inline void Unlock() { mu_.unlock(); }

 private:
  std::mutex mu_;

  Mutex(const Mutex &) = delete;
  Mutex &operator=(const Mutex &) = delete;
};

class MutexLock {
 public:
  explicit MutexLock(Mutex *mu) : mu_(mu) { mu_->Lock(); }

  ~MutexLock() { mu_->Unlock(); }

 private:
  Mutex *mu_;

  MutexLock(const MutexLock &) = delete;
  MutexLock &operator=(const MutexLock &) = delete;
};

// Currently, we don't use a separate reader lock.
// TODO(kbg): Implement this with std::shared_mutex once C++17 becomes widely
// available.
using ReaderMutexLock = MutexLock;

}  // namespace fst

#endif  // FST_LIB_LOCK_H_
