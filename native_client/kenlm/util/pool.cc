#include "util/pool.hh"

#include "util/scoped.hh"

#include <cstdlib>

#include <algorithm>

namespace util {

Pool::Pool() {
  current_ = NULL;
  current_end_ = NULL;
}

Pool::~Pool() {
  FreeAll();
}

void Pool::FreeAll() {
  for (std::vector<void *>::const_iterator i(free_list_.begin()); i != free_list_.end(); ++i) {
    free(*i);
  }
  free_list_.clear();
  current_ = NULL;
  current_end_ = NULL;
}

void *Pool::More(std::size_t size) {
  std::size_t amount = std::max(static_cast<size_t>(32) << free_list_.size(), size);
  uint8_t *ret = static_cast<uint8_t*>(MallocOrThrow(amount));
  free_list_.push_back(ret);
  current_ = ret + size;
  current_end_ = ret + amount;
  return ret;
}

} // namespace util
