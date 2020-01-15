#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#include "util/file.hh"

#include "util/exception.hh"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>

#include <cassert>
#include <cerrno>
#include <climits>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>

#if defined(__MINGW32__)
#include <windows.h>
#include <unistd.h>
#warning "The file functions on MinGW have not been tested for file sizes above 2^31 - 1.  Please read https://stackoverflow.com/questions/12539488/determine-64-bit-file-size-in-c-on-mingw-32-bit and fix"
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace util {

scoped_fd::~scoped_fd() {
  if (fd_ != -1 && close(fd_)) {
    std::cerr << "Could not close file " << fd_ << std::endl;
    std::abort();
  }
}

void scoped_FILE_closer::Close(std::FILE *file) {
  if (file && std::fclose(file)) {
    std::cerr << "Could not close file " << file << std::endl;
    std::abort();
  }
}

// Note that ErrnoException records errno before NameFromFD is called.
FDException::FDException(int fd) throw() : fd_(fd), name_guess_(NameFromFD(fd)) {
  *this << "in " << name_guess_ << ' ';
}

FDException::~FDException() throw() {}

EndOfFileException::EndOfFileException() throw() {
  *this << "End of file";
}
EndOfFileException::~EndOfFileException() throw() {}

bool InputFileIsStdin(StringPiece path) {
  return path == "-" || path == "/dev/stdin";
}

bool OutputFileIsStdout(StringPiece path) {
  return path == "-" || path == "/dev/stdout";
}

int OpenReadOrThrow(const char *name) {
  int ret;
#if defined(_WIN32) || defined(_WIN64)
  UTIL_THROW_IF(-1 == (ret = _open(name, _O_BINARY | _O_RDONLY)), ErrnoException, "while opening " << name);
#else
  UTIL_THROW_IF(-1 == (ret = open(name, O_RDONLY)), ErrnoException, "while opening " << name);
#endif
  return ret;
}

int CreateOrThrow(const char *name) {
  int ret;
#if defined(_WIN32) || defined(_WIN64)
  UTIL_THROW_IF(-1 == (ret = _open(name, _O_CREAT | _O_TRUNC | _O_RDWR | _O_BINARY, _S_IREAD | _S_IWRITE)), ErrnoException, "while creating " << name);
#else
  UTIL_THROW_IF(-1 == (ret = open(name, O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)), ErrnoException, "while creating " << name);
#endif
  return ret;
}

uint64_t SizeFile(int fd) {
#if defined __MINGW32__
  struct stat sb;
  // Does this handle 64-bit?
  int ret = fstat(fd, &sb);
  if (ret == -1 || (!sb.st_size && !S_ISREG(sb.st_mode))) return kBadSize;
  return sb.st_size;
#elif defined(_WIN32) || defined(_WIN64)
  __int64 ret = _filelengthi64(fd);
  return (ret == -1) ? kBadSize : ret;
#else // Not windows.

#ifdef OS_ANDROID
  struct stat64 sb;
  int ret = fstat64(fd, &sb);
#else
  struct stat sb;
  int ret = fstat(fd, &sb);
#endif
  if (ret == -1 || (!sb.st_size && !S_ISREG(sb.st_mode))) return kBadSize;
  return sb.st_size;
#endif
}

uint64_t SizeOrThrow(int fd) {
  uint64_t ret = SizeFile(fd);
  UTIL_THROW_IF_ARG(ret == kBadSize, FDException, (fd), "Failed to size");
  return ret;
}

void ResizeOrThrow(int fd, uint64_t to) {
#if defined __MINGW32__
    // Does this handle 64-bit?
    int ret = ftruncate
#elif defined(_WIN32) || defined(_WIN64)
    errno_t ret = _chsize_s
#elif defined(OS_ANDROID)
    int ret = ftruncate64
#else
    int ret = ftruncate
#endif
    (fd, to);
  UTIL_THROW_IF_ARG(ret, FDException, (fd), "while resizing to " << to << " bytes");
}

void HolePunch(int fd, uint64_t offset, uint64_t size) {
#if defined(__linux__) && defined(FALLOC_FL_PUNCH_HOLE) && defined(FALLOC_FL_KEEP_SIZE)
  UTIL_THROW_IF_ARG(-1 == fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, offset, size), FDException, (fd), "in punching a hole at " << offset << " for " << size << " bytes.");
#else
  UTIL_THROW(UnsupportedOSException, "fallocate hole punching requires Linux and glibc >= 2.18");
#endif
}

namespace {
std::size_t GuardLarge(std::size_t size) {
  // The following operating systems have broken read/write/pread/pwrite that
  // only supports up to 2^31.
  // OS X man pages claim to support 64-bit, but Kareem M. Darwish had problems
  // building with larger files, so APPLE is also here.
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__) || defined(OS_ANDROID) || defined(__MINGW32__)
  return size < INT_MAX ? size : INT_MAX;
#else
  return size;
#endif
}
}

#if defined(_WIN32) || defined(_WIN64)
namespace {
const std::size_t kMaxDWORD = static_cast<std::size_t>(4294967295UL);
} // namespace
#endif

std::size_t PartialRead(int fd, void *to, std::size_t amount) {
#if defined(_WIN32) || defined(_WIN64)
    DWORD ret;
    HANDLE file_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    DWORD larger_size = static_cast<DWORD>(std::min<std::size_t>(kMaxDWORD, amount));
    DWORD smaller_size = 28672; // Received reports that 31346 worked but higher values did not. This rounds down to the nearest multiple of 4096, the page size.
    if (!ReadFile(file_handle, to, larger_size, &ret, NULL))
    {
        DWORD last_error = GetLastError();
        if (last_error != ERROR_NOT_ENOUGH_MEMORY || !ReadFile(file_handle, to, smaller_size, &ret, NULL)) {
            UTIL_THROW(WindowsException, "Windows error in ReadFile.");
        }
    }
#else
  errno = 0;
  ssize_t ret;
  do {
    ret = read(fd, to, GuardLarge(amount));
  } while (ret == -1 && errno == EINTR);
  UTIL_THROW_IF_ARG(ret < 0, FDException, (fd), "while reading " << amount << " bytes");
#endif
  return static_cast<std::size_t>(ret);
}

void ReadOrThrow(int fd, void *to_void, std::size_t amount) {
  uint8_t *to = static_cast<uint8_t*>(to_void);
  while (amount) {
    std::size_t ret = PartialRead(fd, to, amount);
    UTIL_THROW_IF(ret == 0, EndOfFileException, " in " << NameFromFD(fd) << " but there should be " << amount << " more bytes to read.");
    amount -= ret;
    to += ret;
  }
}

std::size_t ReadOrEOF(int fd, void *to_void, std::size_t amount) {
  uint8_t *to = static_cast<uint8_t*>(to_void);
  std::size_t remaining = amount;
  while (remaining) {
    std::size_t ret = PartialRead(fd, to, remaining);
    if (!ret) return amount - remaining;
    remaining -= ret;
    to += ret;
  }
  return amount;
}

void WriteOrThrow(int fd, const void *data_void, std::size_t size) {
  const uint8_t *data = static_cast<const uint8_t*>(data_void);
  while (size) {
#if defined(_WIN32) || defined(_WIN64)
    int ret;
#else
    ssize_t ret;
#endif
    errno = 0;
    do {
      ret =
#if defined(_WIN32) || defined(_WIN64)
        _write
#else
        write
#endif
        (fd, data, GuardLarge(size));
    } while (ret == -1 && errno == EINTR);
    UTIL_THROW_IF_ARG(ret < 1, FDException, (fd), "while writing " << size << " bytes");
    data += ret;
    size -= ret;
  }
}

void WriteOrThrow(FILE *to, const void *data, std::size_t size) {
  if (!size) return;
  UTIL_THROW_IF(1 != std::fwrite(data, size, 1, to), ErrnoException, "Short write; requested size " << size);
}

void ErsatzPRead(int fd, void *to_void, std::size_t size, uint64_t off) {
  uint8_t *to = static_cast<uint8_t*>(to_void);
  while (size) {
#if defined(_WIN32) || defined(_WIN64)
    /* BROKEN: changes file pointer.  Even if you save it and change it back, it won't be safe to use concurrently with write() or read() which lmplz does. */
    // size_t might be 64-bit.  DWORD is always 32.
    DWORD reading = static_cast<DWORD>(std::min<std::size_t>(kMaxDWORD, size));
    DWORD ret;
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(OVERLAPPED));
    overlapped.Offset = static_cast<DWORD>(off);
    overlapped.OffsetHigh = static_cast<DWORD>(off >> 32);
    UTIL_THROW_IF(!ReadFile((HANDLE)_get_osfhandle(fd), to, reading, &ret, &overlapped), WindowsException, "ReadFile failed for offset " << off);
#else
    ssize_t ret;
    errno = 0;
    ret =
#ifdef OS_ANDROID
      pread64
#else
      pread
#endif
      (fd, to, GuardLarge(size), off);
    if (ret <= 0) {
      if (ret == -1 && errno == EINTR) continue;
      UTIL_THROW_IF(ret == 0, EndOfFileException, " for reading " << size << " bytes at " << off << " from " << NameFromFD(fd));
      UTIL_THROW_ARG(FDException, (fd), "while reading " << size << " bytes at offset " << off);
    }
#endif
    size -= ret;
    off += ret;
    to += ret;
  }
}

void ErsatzPWrite(int fd, const void *from_void, std::size_t size, uint64_t off) {
  const uint8_t *from = static_cast<const uint8_t*>(from_void);
  while(size) {
#if defined(_WIN32) || defined(_WIN64)
    /* Changes file pointer.  Even if you save it and change it back, it won't be safe to use concurrently with write() or read() */
    // size_t might be 64-bit.  DWORD is always 32.
    DWORD writing = static_cast<DWORD>(std::min<std::size_t>(kMaxDWORD, size));
    DWORD ret;
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(OVERLAPPED));
    overlapped.Offset = static_cast<DWORD>(off);
    overlapped.OffsetHigh = static_cast<DWORD>(off >> 32);
    UTIL_THROW_IF(!WriteFile((HANDLE)_get_osfhandle(fd), from, writing, &ret, &overlapped), Exception, "WriteFile failed for offset " << off);
#else
    ssize_t ret;
    errno = 0;
    ret =
#ifdef OS_ANDROID
      pwrite64
#else
      pwrite
#endif
      (fd, from, GuardLarge(size), off);
    if (ret <= 0) {
      if (ret == -1 && errno == EINTR) continue;
      UTIL_THROW_IF(ret == 0, EndOfFileException, " for writing " << size << " bytes at " << off << " from " << NameFromFD(fd));
      UTIL_THROW_ARG(FDException, (fd), "while writing " << size << " bytes at offset " << off);
    }
#endif
    size -= ret;
    off += ret;
    from += ret;
  }
}


void FSyncOrThrow(int fd) {
// Apparently windows doesn't have fsync?
#if !defined(_WIN32) && !defined(_WIN64)
  UTIL_THROW_IF_ARG(-1 == fsync(fd), FDException, (fd), "while syncing");
#endif
}

namespace {

// Static assert for 64-bit off_t size.
#if !defined(_WIN32) && !defined(_WIN64) && !defined(OS_ANDROID)
template <unsigned> struct CheckOffT;
template <> struct CheckOffT<8> {
  struct True {};
};
// If there's a compiler error on the next line, then off_t isn't 64 bit.  And
// that makes me a sad panda.
typedef CheckOffT<sizeof(off_t)>::True IgnoredType;
#endif

// Can't we all just get along?
uint64_t InternalSeek(int fd, int64_t off, int whence) {
#if defined __MINGW32__
  // Does this handle 64-bit?
  typedef off_t Offset;
  Offset ret = lseek(fd, off, whence);
#elif defined(_WIN32) || defined(_WIN64)
  typedef __int64 Offset;
  Offset ret = _lseeki64(fd, off, whence);
#elif defined(OS_ANDROID)
  typedef off64_t Offset;
  Offset ret = lseek64(fd, off, whence);
#else
  typedef off_t Offset;
  Offset ret = lseek(fd, off, whence);
#endif
  UTIL_THROW_IF_ARG((Offset)-1 == ret, FDException, (fd), "while seeking to " << off << " whence " << whence);
  return (uint64_t)ret;
}
} // namespace

uint64_t SeekOrThrow(int fd, uint64_t off) {
  return InternalSeek(fd, off, SEEK_SET);
}

uint64_t AdvanceOrThrow(int fd, int64_t off) {
  return InternalSeek(fd, off, SEEK_CUR);
}

uint64_t SeekEnd(int fd) {
  return InternalSeek(fd, 0, SEEK_END);
}

std::FILE *FDOpenOrThrow(scoped_fd &file) {
  std::FILE *ret = fdopen(file.get(), "r+b");
  UTIL_THROW_IF_ARG(!ret, FDException, (file.get()), "Could not fdopen for write");
  file.release();
  return ret;
}

std::FILE *FDOpenReadOrThrow(scoped_fd &file) {
  std::FILE *ret = fdopen(file.get(), "rb");
  UTIL_THROW_IF_ARG(!ret, FDException, (file.get()), "Could not fdopen for read");
  file.release();
  return ret;
}

// Sigh.  Windows temporary file creation is full of race conditions.
#if defined(_WIN32) || defined(_WIN64)
/* mkstemp extracted from libc/sysdeps/posix/tempname.c.  Copyright
   (C) 1991-1999, 2000, 2001, 2006 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.  */

/* This has been modified from the original version to rename the function and
 * set the Windows temporary flag. */

static const char letters[] =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/* Generate a temporary file name based on TMPL.  TMPL must match the
   rules for mk[s]temp (i.e. end in "XXXXXX").  The name constructed
   does not exist at the time of the call to mkstemp.  TMPL is
   overwritten with the result.  */
int
mkstemp_and_unlink(char *tmpl)
{
  int len;
  char *XXXXXX;
  static unsigned long long value;
  unsigned long long random_time_bits;
  unsigned int count;
  int fd = -1;
  int save_errno = errno;

  /* A lower bound on the number of temporary files to attempt to
     generate.  The maximum total number of temporary file names that
     can exist for a given template is 62**6.  It should never be
     necessary to try all these combinations.  Instead if a reasonable
     number of names is tried (we define reasonable as 62**3) fail to
     give the system administrator the chance to remove the problems.  */
#define ATTEMPTS_MIN (62 * 62 * 62)

  /* The number of times to attempt to generate a temporary file.  To
     conform to POSIX, this must be no smaller than TMP_MAX.  */
#if ATTEMPTS_MIN < TMP_MAX
  unsigned int attempts = TMP_MAX;
#else
  unsigned int attempts = ATTEMPTS_MIN;
#endif

  len = strlen (tmpl);
  if (len < 6 || strcmp (&tmpl[len - 6], "XXXXXX"))
    {
      errno = EINVAL;
      return -1;
    }

/* This is where the Xs start.  */
  XXXXXX = &tmpl[len - 6];

  /* Get some more or less random data.  */
  {
    SYSTEMTIME      stNow;
    FILETIME ftNow;

    // get system time
    GetSystemTime(&stNow);
    stNow.wMilliseconds = 500;
    if (!SystemTimeToFileTime(&stNow, &ftNow))
    {
        errno = -1;
        return -1;
    }

    random_time_bits = (((unsigned long long)ftNow.dwHighDateTime << 32)
                        | (unsigned long long)ftNow.dwLowDateTime);
  }
  value += random_time_bits ^ (unsigned long long)GetCurrentThreadId ();

  for (count = 0; count < attempts; value += 7777, ++count)
  {
    unsigned long long v = value;

    /* Fill in the random bits.  */
    XXXXXX[0] = letters[v % 62];
    v /= 62;
    XXXXXX[1] = letters[v % 62];
    v /= 62;
    XXXXXX[2] = letters[v % 62];
    v /= 62;
    XXXXXX[3] = letters[v % 62];
    v /= 62;
    XXXXXX[4] = letters[v % 62];
    v /= 62;
    XXXXXX[5] = letters[v % 62];

    /* Modified for windows and to unlink */
    //      fd = open (tmpl, O_RDWR | O_CREAT | O_EXCL, _S_IREAD | _S_IWRITE);
    int flags = _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY;
    flags |= _O_TEMPORARY;
    fd = _open (tmpl, flags, _S_IREAD | _S_IWRITE);
    if (fd >= 0)
    {
      errno = save_errno;
      return fd;
    }
    else if (errno != EEXIST)
      return -1;
  }

  /* We got out of the loop because we ran out of combinations to try.  */
  errno = EEXIST;
  return -1;
}
#else
int
mkstemp_and_unlink(char *tmpl) {
  int ret = mkstemp(tmpl);
  if (ret != -1) {
    UTIL_THROW_IF(unlink(tmpl), ErrnoException, "while deleting " << tmpl);
  }
  return ret;
}
#endif

// If it's a directory, add a /.  This lets users say -T /tmp without creating
// /tmpAAAAAA
void NormalizeTempPrefix(std::string &base) {
  if (base.empty()) return;
  if (base[base.size() - 1] == '/') return;
  struct stat sb;
  // It's fine for it to not exist.
  if (-1 == stat(base.c_str(), &sb)) return;
  if (
#if defined(_WIN32) || defined(_WIN64)
    sb.st_mode & _S_IFDIR
#else
    S_ISDIR(sb.st_mode)
#endif
    ) base += '/';
}

int MakeTemp(const StringPiece &base) {
  std::string name(base.data(), base.size());
  name += "XXXXXX";
  name.push_back(0);
  int ret;
  UTIL_THROW_IF(-1 == (ret = mkstemp_and_unlink(&name[0])), ErrnoException, "while making a temporary based on " << base);
  return ret;
}

std::FILE *FMakeTemp(const StringPiece &base) {
  util::scoped_fd file(MakeTemp(base));
  return FDOpenOrThrow(file);
}

std::string DefaultTempDirectory() {
#if defined(_WIN32) || defined(_WIN64)
  char dir_buffer[1000];
  if (GetTempPath(1000, dir_buffer) == 0)
    throw std::runtime_error("Could not read temporary directory.");
  std::string ret(dir_buffer);
  NormalizeTempPrefix(ret);
  return ret;
#else
  // POSIX says to try these environment variables, in this order:
  const char *const vars[] = {"TMPDIR", "TMP", "TEMPDIR", "TEMP", 0};
  for (int i=0; vars[i]; ++i) {
    char *val =
#if defined(_GNU_SOURCE) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2,17)
      secure_getenv
#else // __GLIBC_PREREQ
      getenv
#endif // __GLIBC_PREREQ
#else // _GNU_SOURCE
      getenv
#endif
      (vars[i]);
    // Environment variable is set and nonempty.  Use it.
    if (val && *val) {
      std::string ret(val);
      NormalizeTempPrefix(ret);
      return ret;
    }
  }
  // No environment variables set.  Default to /tmp.
  return "/tmp/";
#endif
}

int DupOrThrow(int fd) {
  int ret = dup(fd);
  UTIL_THROW_IF_ARG(ret == -1, FDException, (fd), "in duplicating the file descriptor");
  return ret;
}

namespace {
// Try to name things but be willing to fail too.
bool TryName(int fd, std::string &out) {
#if defined(_WIN32) || defined(_WIN64)
  return false;
#else
  std::string name("/proc/self/fd/");
  std::ostringstream convert;
  convert << fd;
  name += convert.str();

  struct stat sb;
  if (-1 == lstat(name.c_str(), &sb))
    return false;
  out.resize(sb.st_size + 1);
  // lstat gave us a size, but I've seen it grow, possibly due to symlinks on top of symlinks.
  while (true) {
    ssize_t ret = readlink(name.c_str(), &out[0], out.size());
    if (-1 == ret)
      return false;
    if ((size_t)ret < out.size()) {
      out.resize(ret);
      break;
    }
    // Exponential growth.
    out.resize(out.size() * 2);
  }
  // Don't use the non-file names.
  if (!out.empty() && out[0] != '/')
    return false;
  return true;
#endif
}
} // namespace

std::string NameFromFD(int fd) {
  std::string ret;
  if (TryName(fd, ret)) return ret;
  switch (fd) {
    case 0: return "stdin";
    case 1: return "stdout";
    case 2: return "stderr";
  }
  ret = "fd ";
  std::ostringstream convert;
  convert << fd;
  ret += convert.str();
  return ret;
}

} // namespace util
