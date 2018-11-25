#include "util/scoped.hh"

#include <cstdlib>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/mman.h>
#endif

namespace util {

// TODO: if we're really under memory pressure, don't allocate memory to
// display the error.
MallocException::MallocException(std::size_t requested) throw() {
  *this << "for " << requested << " bytes ";
}

MallocException::~MallocException() throw() {}

namespace {
void *InspectAddr(void *addr, std::size_t requested, const char *func_name) {
  UTIL_THROW_IF_ARG(!addr && requested, MallocException, (requested), "in " << func_name);
  return addr;
}
} // namespace

void *MallocOrThrow(std::size_t requested) {
  return InspectAddr(std::malloc(requested), requested, "malloc");
}

void *CallocOrThrow(std::size_t requested) {
  return InspectAddr(std::calloc(requested, 1), requested, "calloc");
}

void scoped_malloc::call_realloc(std::size_t requested) {
  p_ = InspectAddr(std::realloc(p_, requested), requested, "realloc");
}

void AdviseHugePages(const void *addr, std::size_t size) {
#if MADV_HUGEPAGE
  madvise((void*)addr, size, MADV_HUGEPAGE);
#endif
}

} // namespace util
