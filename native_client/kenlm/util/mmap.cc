/* Memory mapping wrappers.
 * ARM and MinGW ports contributed by Hideo Okuma and Tomoyuki Yoshimura at
 * NICT.
 */
#include "util/mmap.hh"

#include "util/exception.hh"
#include "util/file.hh"
#include "util/parallel_read.hh"
#include "util/scoped.hh"

#include <iostream>

#include <cassert>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdlib>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace util {

std::size_t SizePage() {
#if defined(_WIN32) || defined(_WIN64)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwAllocationGranularity;
#else
  return sysconf(_SC_PAGE_SIZE);
#endif
}

scoped_mmap::~scoped_mmap() {
  if (data_ != (void*)-1) {
    try {
      // Thanks Denis Filimonov for pointing out NFS likes msync first.
      SyncOrThrow(data_, size_);
      UnmapOrThrow(data_, size_);
    } catch (const util::ErrnoException &e) {
      std::cerr << e.what();
      abort();
    }
  }
}

namespace {
template <class T> T RoundUpPow2(T value, T mult) {
  return ((value - 1) & ~(mult - 1)) + mult;
}
} // namespace

scoped_memory::scoped_memory(std::size_t size, bool zeroed) : data_(NULL), size_(0), source_(NONE_ALLOCATED) {
  HugeMalloc(size, zeroed, *this);
}

void scoped_memory::reset(void *data, std::size_t size, Alloc source) {
  switch(source_) {
    case MMAP_ROUND_UP_ALLOCATED:
      scoped_mmap(data_, RoundUpPow2(size_, (std::size_t)SizePage()));
      break;
    case MMAP_ALLOCATED:
      scoped_mmap(data_, size_);
      break;
    case MALLOC_ALLOCATED:
      free(data_);
      break;
    case NONE_ALLOCATED:
      break;
  }
  data_ = data;
  size_ = size;
  source_ = source;
}

/*void scoped_memory::call_realloc(std::size_t size) {
  assert(source_ == MALLOC_ALLOCATED || source_ == NONE_ALLOCATED);
  void *new_data = realloc(data_, size);
  if (!new_data) {
    reset();
  } else {
    data_ = new_data;
    size_ = size;
    source_ = MALLOC_ALLOCATED;
  }
}*/

const int kFileFlags =
#if defined(_WIN32) || defined(_WIN64)
  0 // MapOrThrow ignores flags on windows
#elif defined(MAP_FILE)
  MAP_FILE | MAP_SHARED
#else
  MAP_SHARED
#endif
  ;

void *MapOrThrow(std::size_t size, bool for_write, int flags, bool prefault, int fd, uint64_t offset) {
#ifdef MAP_POPULATE // Linux specific
  if (prefault) {
    flags |= MAP_POPULATE;
  }
#endif
#if defined(_WIN32) || defined(_WIN64)
  int protectC = for_write ? PAGE_READWRITE : PAGE_READONLY;
  int protectM = for_write ? FILE_MAP_WRITE : FILE_MAP_READ;
  uint64_t total_size = size + offset;
  HANDLE hMapping = CreateFileMapping((HANDLE)_get_osfhandle(fd), NULL, protectC, total_size >> 32, static_cast<DWORD>(total_size), NULL);
  UTIL_THROW_IF(!hMapping, ErrnoException, "CreateFileMapping failed");
  LPVOID ret = MapViewOfFile(hMapping, protectM, offset >> 32, offset, size);
  CloseHandle(hMapping);
  UTIL_THROW_IF(!ret, ErrnoException, "MapViewOfFile failed");
#else
  int protect = for_write ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void *ret;
  UTIL_THROW_IF((ret = mmap(NULL, size, protect, flags, fd, offset)) == MAP_FAILED, ErrnoException, "mmap failed for size " << size << " at offset " << offset);
#  ifdef MADV_HUGEPAGE
  /* We like huge pages but it's fine if we can't have them.  Note that huge
   * pages are not supported for file-backed mmap on linux.
   */
  madvise(ret, size, MADV_HUGEPAGE);
#  endif
#endif
  return ret;
}

void SyncOrThrow(void *start, size_t length) {
#if defined(_WIN32) || defined(_WIN64)
  UTIL_THROW_IF(!::FlushViewOfFile(start, length), ErrnoException, "Failed to sync mmap");
#else
  UTIL_THROW_IF(length && msync(start, length, MS_SYNC), ErrnoException, "Failed to sync mmap");
#endif
}

void UnmapOrThrow(void *start, size_t length) {
#if defined(_WIN32) || defined(_WIN64)
  UTIL_THROW_IF(!::UnmapViewOfFile(start), ErrnoException, "Failed to unmap a file");
#else
  UTIL_THROW_IF(munmap(start, length), ErrnoException, "munmap failed");
#endif
}

// Linux huge pages.
#ifdef __linux__

namespace {

bool AnonymousMap(std::size_t size, int flags, bool populate, util::scoped_memory &to) {
  if (populate) flags |= MAP_POPULATE;
  void *ret = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | flags, -1, 0);
  if (ret == MAP_FAILED) return false;
  to.reset(ret, size, scoped_memory::MMAP_ALLOCATED);
  return true;
}

bool TryHuge(std::size_t size, uint8_t alignment_bits, bool populate, util::scoped_memory &to) {
  // Don't bother with these cases.
  if (size < (1ULL << alignment_bits) || (1ULL << alignment_bits) < SizePage())
    return false;

  // First try: Linux >= 3.8 with manually configured hugetlb pages available.
#ifdef MAP_HUGE_SHIFT
  if (AnonymousMap(size, MAP_HUGETLB | (alignment_bits << MAP_HUGE_SHIFT), populate, to))
    return true;
#endif

  // Second try: manually configured hugetlb pages exist, but kernel too old to
  // pick size or not available.  This might pick the wrong size huge pages,
  // but the sysadmin must have made them available in the first place.
  if (AnonymousMap(size, MAP_HUGETLB, populate, to))
    return true;

  // Third try: align to a multiple of the huge page size by overallocating.
  // I feel bad about doing this, but it's also how posix_memalign is
  // implemented.  And the memory is virtual.

  // Round up requested size to multiple of page size.  This will allow the pages after to be munmapped.
  std::size_t size_up = RoundUpPow2(size, SizePage());

  std::size_t ask = size_up + (1 << alignment_bits) - SizePage();
  // Don't populate because this is asking for more than we will use.
  scoped_mmap larger(mmap(NULL, ask, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), ask);
  if (larger.get() == MAP_FAILED) return false;

  // Throw out pages before the alignment point.
  uintptr_t base = reinterpret_cast<uintptr_t>(larger.get());
  // Round up to next multiple of alignment.
  uintptr_t rounded_up = RoundUpPow2(base, static_cast<uintptr_t>(1) << alignment_bits);
  if (base != rounded_up) {
    // If this throws an exception (which it shouldn't) then we want to unmap the whole thing by keeping it in larger.
    UnmapOrThrow(larger.get(), rounded_up - base);
    larger.steal();
    larger.reset(reinterpret_cast<void*>(rounded_up), ask - (rounded_up - base));
  }

  // Throw out pages after the requested size.
  assert(larger.size() >= size_up);
  if (larger.size() > size_up) {
    // This is where we assume size_up is a multiple of page size.
    UnmapOrThrow(static_cast<uint8_t*>(larger.get()) + size_up, larger.size() - size_up);
    larger.reset(larger.steal(), size_up);
  }
#ifdef MADV_HUGEPAGE
  madvise(larger.get(), size_up, MADV_HUGEPAGE);
#endif
  to.reset(larger.steal(), size, scoped_memory::MMAP_ROUND_UP_ALLOCATED);
  return true;
}

} // namespace

#endif

void HugeMalloc(std::size_t size, bool zeroed, scoped_memory &to) {
  to.reset();
#ifdef __linux__
  // TODO: architectures/page sizes other than 2^21 and 2^30.
  // Attempt 1 GB pages.
  // If the user asked for zeroed memory, assume they want it populated.
  if (size >= (1ULL << 30) && TryHuge(size, 30, zeroed, to))
    return;
  // Attempt 2 MB pages.
  if (size >= (1ULL << 21) && TryHuge(size, 21, zeroed, to))
    return;
#endif // __linux__
  // Non-linux will always do this, as will small allocations on Linux.
  to.reset(zeroed ? calloc(1, size) : malloc(size), size, scoped_memory::MALLOC_ALLOCATED);
  UTIL_THROW_IF(!to.get(), ErrnoException, "Failed to allocate " << size << " bytes");
}

#ifdef __linux__
const std::size_t kTransitionHuge = std::max<std::size_t>(1ULL << 21, SizePage());
#endif // __linux__

void HugeRealloc(std::size_t to, bool zero_new, scoped_memory &mem) {
  if (!to) {
    mem.reset();
    return;
  }
#ifdef __linux__
  std::size_t from_size = mem.size();
#endif // __linux__
  switch (mem.source()) {
    case scoped_memory::NONE_ALLOCATED:
      HugeMalloc(to, zero_new, mem);
      return;
#ifdef __linux__
    case scoped_memory::MMAP_ROUND_UP_ALLOCATED:
      // for mremap's benefit.
      from_size = RoundUpPow2(from_size, SizePage());
    case scoped_memory::MMAP_ALLOCATED:
      // Downsizing below barrier?
      if (to <= SizePage()) {
        scoped_malloc replacement(malloc(to));
        memcpy(replacement.get(), mem.get(), std::min(to, mem.size()));
        if (zero_new && to > mem.size())
          memset(static_cast<uint8_t*>(replacement.get()) + mem.size(), 0, to - mem.size());
        mem.reset(replacement.release(), to, scoped_memory::MALLOC_ALLOCATED);
      } else {
        void *new_addr = mremap(mem.get(), from_size, to, MREMAP_MAYMOVE);
        UTIL_THROW_IF(!new_addr, ErrnoException, "Failed to mremap from " << from_size << " to " << to);
        mem.steal();
        mem.reset(new_addr, to, scoped_memory::MMAP_ALLOCATED);
      }
      return;
#endif // __linux__
    case scoped_memory::MALLOC_ALLOCATED:
#ifdef __linux__
      // Transition larger allocations to huge pages, but don't keep trying if we're still malloc allocated.
      if (to >= kTransitionHuge && mem.size() < kTransitionHuge) {
        scoped_memory replacement;
        HugeMalloc(to, zero_new, replacement);
        memcpy(replacement.get(), mem.get(), mem.size());
        // This can't throw.
        mem.reset(replacement.get(), replacement.size(), replacement.source());
        replacement.steal();
        return;
      }
#endif // __linux__
      {
        void *new_addr = std::realloc(mem.get(), to);
        UTIL_THROW_IF(!new_addr, ErrnoException, "realloc to " << to << " bytes failed.");
        if (zero_new && to > mem.size())
          memset(static_cast<uint8_t*>(new_addr) + mem.size(), 0, to - mem.size());
        mem.steal();
        mem.reset(new_addr, to, scoped_memory::MALLOC_ALLOCATED);
      }
      return;
    default:
      UTIL_THROW(Exception, "HugeRealloc called with type " << mem.source());
  }
}

void MapRead(LoadMethod method, int fd, uint64_t offset, std::size_t size, scoped_memory &out) {
  switch (method) {
    case LAZY:
      out.reset(MapOrThrow(size, false, kFileFlags, false, fd, offset), size, scoped_memory::MMAP_ALLOCATED);
      break;
    case POPULATE_OR_LAZY:
#ifdef MAP_POPULATE
    case POPULATE_OR_READ:
#endif
      out.reset(MapOrThrow(size, false, kFileFlags, true, fd, offset), size, scoped_memory::MMAP_ALLOCATED);
      break;
#ifndef MAP_POPULATE
    case POPULATE_OR_READ:
#endif
    case READ:
      HugeMalloc(size, false, out);
      SeekOrThrow(fd, offset);
      ReadOrThrow(fd, out.get(), size);
      break;
    case PARALLEL_READ:
      HugeMalloc(size, false, out);
      ParallelRead(fd, out.get(), size, offset);
      break;
  }
}

void *MapZeroedWrite(int fd, std::size_t size) {
  ResizeOrThrow(fd, 0);
  ResizeOrThrow(fd, size);
  return MapOrThrow(size, true, kFileFlags, false, fd, 0);
}

void *MapZeroedWrite(const char *name, std::size_t size, scoped_fd &file) {
  file.reset(CreateOrThrow(name));
  try {
    return MapZeroedWrite(file.get(), size);
  } catch (ErrnoException &e) {
    e << " in file " << name;
    throw;
  }
}

Rolling::Rolling(const Rolling &copy_from, uint64_t increase) {
  *this = copy_from;
  IncreaseBase(increase);
}

Rolling &Rolling::operator=(const Rolling &copy_from) {
  fd_ = copy_from.fd_;
  file_begin_ = copy_from.file_begin_;
  file_end_ = copy_from.file_end_;
  for_write_ = copy_from.for_write_;
  block_ = copy_from.block_;
  read_bound_ = copy_from.read_bound_;

  current_begin_ = 0;
  if (copy_from.IsPassthrough()) {
    current_end_ = copy_from.current_end_;
    ptr_ = copy_from.ptr_;
  } else {
    // Force call on next mmap.
    current_end_ = 0;
    ptr_ = NULL;
  }
  return *this;
}

Rolling::Rolling(int fd, bool for_write, std::size_t block, std::size_t read_bound, uint64_t offset, uint64_t amount) {
  current_begin_ = 0;
  current_end_ = 0;
  fd_ = fd;
  file_begin_ = offset;
  file_end_ = offset + amount;
  for_write_ = for_write;
  block_ = block;
  read_bound_ = read_bound;
}

void *Rolling::ExtractNonRolling(scoped_memory &out, uint64_t index, std::size_t size) {
  out.reset();
  if (IsPassthrough()) return static_cast<uint8_t*>(get()) + index;
  uint64_t offset = index + file_begin_;
  // Round down to multiple of page size.
  uint64_t cruft = offset % static_cast<uint64_t>(SizePage());
  std::size_t map_size = static_cast<std::size_t>(size + cruft);
  out.reset(MapOrThrow(map_size, for_write_, kFileFlags, true, fd_, offset - cruft), map_size, scoped_memory::MMAP_ALLOCATED);
  return static_cast<uint8_t*>(out.get()) + static_cast<std::size_t>(cruft);
}

void Rolling::Roll(uint64_t index) {
  assert(!IsPassthrough());
  std::size_t amount;
  if (file_end_ - (index + file_begin_) > static_cast<uint64_t>(block_)) {
    amount = block_;
    current_end_ = index + amount - read_bound_;
  } else {
    amount = file_end_ - (index + file_begin_);
    current_end_ = index + amount;
  }
  ptr_ = static_cast<uint8_t*>(ExtractNonRolling(mem_, index, amount)) - index;

  current_begin_ = index;
}

} // namespace util
