// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//

#include <fst/mapped-file.h>

#include <errno.h>
#include <fcntl.h>
#ifdef HAVE_SYS_MMAN
#include <sys/mman.h>
#endif  // HAVE_SYS_MMAN
#ifndef _MSC_VER
#include <unistd.h>
#endif  // _MSC_VER

#include <algorithm>
#include <ios>
#include <memory>

#include <fst/log.h>

namespace fst {

MappedFile::MappedFile(const MemoryRegion &region) : region_(region) {}

MappedFile::~MappedFile() {
  if (region_.size != 0) {
#ifdef HAVE_SYS_MMAN
    if (region_.mmap) {
      VLOG(1) << "munmap'ed " << region_.size << " bytes at " << region_.mmap;
      if (munmap(region_.mmap, region_.size) != 0) {
        LOG(ERROR) << "Failed to unmap region: " << strerror(errno);
      }
    } else
#endif  // HAVE_SYS_MMAN
    {
      if (region_.data) {
        operator delete(static_cast<char *>(region_.data) - region_.offset);
      }
    }
  }
}

MappedFile *MappedFile::Map(std::istream *istrm, bool memorymap,
                            const string &source, size_t size) {
  (void)memorymap;
  const auto spos = istrm->tellg();
#ifdef HAVE_SYS_MMAN
  VLOG(1) << "memorymap: " << (memorymap ? "true" : "false") << " source: \""
          << source << "\""
          << " size: " << size << " offset: " << spos;
  if (memorymap && spos >= 0 && spos % kArchAlignment == 0) {
    const size_t pos = spos;
    int fd = open(source.c_str(), O_RDONLY);
    if (fd != -1) {
      const int pagesize = sysconf(_SC_PAGESIZE);
      const off_t offset = pos % pagesize;
      const off_t upsize = size + offset;
      void *map =
          mmap(nullptr, upsize, PROT_READ, MAP_SHARED, fd, pos - offset);
      auto *data = reinterpret_cast<char *>(map);
      if (close(fd) == 0 && map != MAP_FAILED) {
        MemoryRegion region;
        region.mmap = map;
        region.size = upsize;
        region.data = reinterpret_cast<void *>(data + offset);
        region.offset = offset;
        std::unique_ptr<MappedFile> mmf(new MappedFile(region));
        istrm->seekg(pos + size, std::ios::beg);
        if (istrm) {
          VLOG(1) << "mmap'ed region of " << size << " at offset " << pos
                  << " from " << source << " to addr " << map;
          return mmf.release();
        }
      } else {
        LOG(INFO) << "Mapping of file failed: " << strerror(errno);
      }
    }
  }
  // If all else fails, reads from the file into the allocated buffer.
  if (memorymap) {
    LOG(WARNING) << "File mapping at offset " << spos << " of file " << source
                 << " could not be honored, reading instead";
  }
#endif  // HAVE_SYS_MMAN

  // Reads the file into the buffer in chunks not larger than kMaxReadChunk.
  std::unique_ptr<MappedFile> mf(Allocate(size));
  auto *buffer = reinterpret_cast<char *>(mf->mutable_data());
  while (size > 0) {
    const auto next_size = std::min(size, kMaxReadChunk);
    const auto current_pos = istrm->tellg();
    if (!istrm->read(buffer, next_size)) {
      LOG(ERROR) << "Failed to read " << next_size << " bytes at offset "
                 << current_pos << "from \"" << source << "\"";
      return nullptr;
    }
    size -= next_size;
    buffer += next_size;
    VLOG(2) << "Read " << next_size << " bytes. " << size << " remaining";
  }
  return mf.release();
}

MappedFile *MappedFile::Allocate(size_t size, int align) {
  MemoryRegion region;
  region.data = nullptr;
  region.offset = 0;
  if (size > 0) {
    char *buffer = static_cast<char *>(operator new(size + align));
    size_t address = reinterpret_cast<size_t>(buffer);
    region.offset = kArchAlignment - (address % align);
    region.data = buffer + region.offset;
  }
  region.mmap = nullptr;
  region.size = size;
  return new MappedFile(region);
}

MappedFile *MappedFile::Borrow(void *data) {
  MemoryRegion region;
  region.data = data;
  region.mmap = data;
  region.size = 0;
  region.offset = 0;
  return new MappedFile(region);
}

constexpr int MappedFile::kArchAlignment;

constexpr size_t MappedFile::kMaxReadChunk;

}  // namespace fst
