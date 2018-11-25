#ifndef UTIL_READ_COMPRESSED_H
#define UTIL_READ_COMPRESSED_H

#include "util/exception.hh"
#include "util/scoped.hh"

#include <cstddef>
#include <stdint.h>

namespace util {

class CompressedException : public Exception {
  public:
    CompressedException() throw();
    virtual ~CompressedException() throw();
};

class GZException : public CompressedException {
  public:
    GZException() throw();
    ~GZException() throw();
};

class BZException : public CompressedException {
  public:
    BZException() throw();
    ~BZException() throw();
};

class XZException : public CompressedException {
  public:
    XZException() throw();
    ~XZException() throw();
};

class ReadCompressed;

class ReadBase {
  public:
    virtual ~ReadBase() {}

    virtual std::size_t Read(void *to, std::size_t amount, ReadCompressed &thunk) = 0;

  protected:
    static void ReplaceThis(ReadBase *with, ReadCompressed &thunk);

    ReadBase *Current(ReadCompressed &thunk);

    static uint64_t &ReadCount(ReadCompressed &thunk);
};

class ReadCompressed {
  public:
    static const std::size_t kMagicSize = 6;
    // Must have at least kMagicSize bytes.
    static bool DetectCompressedMagic(const void *from);

    // Takes ownership of fd.
    explicit ReadCompressed(int fd);

    // Try to avoid using this.  Use the fd instead.
    // There is no decompression support for istreams.
    explicit ReadCompressed(std::istream &in);

    // Must call Reset later.
    ReadCompressed();

    // Takes ownership of fd.
    void Reset(int fd);

    // Same advice as the constructor.
    void Reset(std::istream &in);

    std::size_t Read(void *to, std::size_t amount);

    // Repeatedly call read to fill a buffer unless EOF is hit.
    // Return number of bytes read.
    std::size_t ReadOrEOF(void *const to, std::size_t amount);

    uint64_t RawAmount() const { return raw_amount_; }

  private:
    friend class ReadBase;

    scoped_ptr<ReadBase> internal_;

    uint64_t raw_amount_;
};

} // namespace util

#endif // UTIL_READ_COMPRESSED_H
