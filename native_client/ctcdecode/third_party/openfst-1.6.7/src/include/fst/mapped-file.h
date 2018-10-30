// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_MAPPED_FILE_H_
#define FST_MAPPED_FILE_H_

#include <cstddef>
#include <istream>
#include <string>

#include <fst/compat.h>
#include <fst/flags.h>

namespace fst {

// A memory region is a simple abstraction for allocated memory or data from
// memory-mapped files. If mmap is null, then data represents an owned region
// of size bytes. Otherwise, mmap and size refer to the mapping and data is a
// casted pointer to a region contained within [mmap, mmap + size). If size is
// 0, then mmap and data refer to a block of memory managed externally by some
// other allocator. The offset is used when allocating memory to providing
// padding for alignment.
struct MemoryRegion {
  void *data;
  void *mmap;
  size_t size;
  int offset;
};

class MappedFile {
 public:
  ~MappedFile();

  void *mutable_data() const { return region_.data; }

  const void *data() const { return region_.data; }

  // Returns a MappedFile object that contains the contents of the input stream
  // strm starting from the current file position with size bytes. The memorymap
  // bool is advisory, and Map will default to allocating and reading. The
  // source argument needs to contain the filename that was used to open the
  // input stream.
  static MappedFile *Map(std::istream *istrm, bool memorymap,
                         const string &source, size_t size);

  // Creates a MappedFile object with a new[]'ed block of memory of size. The
  // align argument can be used to specify a desired block alignment.
  // This is RECOMMENDED FOR INTERNAL USE ONLY as it may change in future
  // releases.
  static MappedFile *Allocate(size_t size, int align = kArchAlignment);

  // Creates a MappedFile object pointing to a borrowed reference to data. This
  // block of memory is not owned by the MappedFile object and will not be
  // freed. This is RECOMMENDED FOR INTERNAL USE ONLY, may change in future
  // releases.
  static MappedFile *Borrow(void *data);

  // Alignment required for mapping structures in bytes. Regions of memory that
  // are not aligned upon a 128-bit boundary are read from the file instead.
  // This is consistent with the alignment boundary set in ConstFst and
  // CompactFst.
  static constexpr int kArchAlignment = 16;

  static constexpr size_t kMaxReadChunk = 256 * 1024 * 1024;  // 256 MB.

 private:
  explicit MappedFile(const MemoryRegion &region);

  MemoryRegion region_;
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
};
}  // namespace fst

#endif  // FST_MAPPED_FILE_H_
