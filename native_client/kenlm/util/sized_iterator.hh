#ifndef UTIL_SIZED_ITERATOR_H
#define UTIL_SIZED_ITERATOR_H

#include "util/pool.hh"
#include "util/proxy_iterator.hh"

#include <algorithm>
#include <functional>
#include <string>

#include <stdint.h>
#include <cstring>

#include <stdlib.h>

namespace util {

class SizedInnerIterator {
  public:
    SizedInnerIterator() {}

    SizedInnerIterator(void *ptr, std::size_t size) : ptr_(static_cast<uint8_t*>(ptr)), size_(size) {}

    bool operator==(const SizedInnerIterator &other) const {
      return ptr_ == other.ptr_;
    }
    bool operator<(const SizedInnerIterator &other) const {
      return ptr_ < other.ptr_;
    }
    SizedInnerIterator &operator+=(std::ptrdiff_t amount) {
      ptr_ += amount * size_;
      return *this;
    }
    std::ptrdiff_t operator-(const SizedInnerIterator &other) const {
      return (ptr_ - other.ptr_) / size_;
    }

    const void *Data() const { return ptr_; }
    void *Data() { return ptr_; }
    std::size_t EntrySize() const { return size_; }

    friend void swap(SizedInnerIterator &first, SizedInnerIterator &second);

  private:
    uint8_t *ptr_;
    std::size_t size_;
};

inline void swap(SizedInnerIterator &first, SizedInnerIterator &second) {
  using std::swap;
  swap(first.ptr_, second.ptr_);
  swap(first.size_, second.size_);
}

class ValueBlock {
  public:
    explicit ValueBlock(const void *from, FreePool &pool)
      : ptr_(std::memcpy(pool.Allocate(), from, pool.ElementSize())),
        pool_(pool) {}

    ValueBlock(const ValueBlock &from)
      : ptr_(std::memcpy(from.pool_.Allocate(), from.ptr_, from.pool_.ElementSize())),
        pool_(from.pool_) {}

    ValueBlock &operator=(const ValueBlock &from) {
      std::memcpy(ptr_, from.ptr_, pool_.ElementSize());
      return *this;
    }

    ~ValueBlock() { pool_.Free(ptr_); }

    const void *Data() const { return ptr_; }
    void *Data() { return ptr_; }

  private:
    void *ptr_;
    FreePool &pool_;
};

class SizedProxy {
  public:
    SizedProxy() {}

    SizedProxy(void *ptr, FreePool &pool) : inner_(ptr, pool.ElementSize()), pool_(&pool) {}

    operator ValueBlock() const {
      return ValueBlock(inner_.Data(), *pool_);
    }

    SizedProxy &operator=(const SizedProxy &from) {
      memcpy(inner_.Data(), from.inner_.Data(), inner_.EntrySize());
      return *this;
    }

    SizedProxy &operator=(const ValueBlock &from) {
      memcpy(inner_.Data(), from.Data(), inner_.EntrySize());
      return *this;
    }

    const void *Data() const { return inner_.Data(); }
    void *Data() { return inner_.Data(); }

    friend void swap(SizedProxy first, SizedProxy second);

  private:
    friend class util::ProxyIterator<SizedProxy>;

    typedef ValueBlock value_type;

    typedef SizedInnerIterator InnerIterator;

    InnerIterator &Inner() { return inner_; }
    const InnerIterator &Inner() const { return inner_; }

    InnerIterator inner_;

    FreePool *pool_;
};

inline void swap(SizedProxy first, SizedProxy second) {
  std::swap_ranges(
      static_cast<char*>(first.inner_.Data()),
      static_cast<char*>(first.inner_.Data()) + first.inner_.EntrySize(),
      static_cast<char*>(second.inner_.Data()));
}

typedef ProxyIterator<SizedProxy> SizedIterator;

// Useful wrapper for a comparison function i.e. sort.
template <class Delegate, class Proxy = SizedProxy> class SizedCompare : public std::binary_function<const Proxy &, const Proxy &, bool> {
  public:
    explicit SizedCompare(const Delegate &delegate = Delegate()) : delegate_(delegate) {}

    bool operator()(const Proxy &first, const Proxy &second) const {
      return delegate_(first.Data(), second.Data());
    }
    bool operator()(const Proxy &first, const ValueBlock &second) const {
      return delegate_(first.Data(), second.Data());
    }
    bool operator()(const ValueBlock &first, const Proxy &second) const {
      return delegate_(first.Data(), second.Data());
    }
    bool operator()(const ValueBlock &first, const ValueBlock &second) const {
      return delegate_(first.Data(), second.Data());
    }

    const Delegate &GetDelegate() const { return delegate_; }

  private:
    const Delegate delegate_;
};

template <unsigned Size> class JustPOD {
  unsigned char data[Size];
};

template <class Delegate, unsigned Size> class JustPODDelegate : std::binary_function<const JustPOD<Size> &, const JustPOD<Size> &, bool> {
  public:
    explicit JustPODDelegate(const Delegate &compare) : delegate_(compare) {}
    bool operator()(const JustPOD<Size> &first, const JustPOD<Size> &second) const {
      return delegate_(&first, &second);
    }
  private:
    Delegate delegate_;
};

#define UTIL_SORT_SPECIALIZE(Size) \
  case Size: \
    std::sort(static_cast<JustPOD<Size>*>(start), static_cast<JustPOD<Size>*>(end), JustPODDelegate<Compare, Size>(compare)); \
    break;

template <class Compare> void SizedSort(void *start, void *end, std::size_t element_size, const Compare &compare) {
  switch (element_size) {
    // Benchmarking sort found it's about 2x faster with an explicitly sized type.  So here goes :-(.
    UTIL_SORT_SPECIALIZE(4);
    UTIL_SORT_SPECIALIZE(8);
    UTIL_SORT_SPECIALIZE(12);
    UTIL_SORT_SPECIALIZE(16);
    UTIL_SORT_SPECIALIZE(17); // This is used by interpolation.
    UTIL_SORT_SPECIALIZE(20);
    UTIL_SORT_SPECIALIZE(24);
    UTIL_SORT_SPECIALIZE(28);
    UTIL_SORT_SPECIALIZE(32);
    default:
      // Recent g++ versions create a temporary value_type then compare with it.
      // Problem is that value_type in this case needs to be a runtime-sized array.
      // Previously I had std::string serve this role.  However, there were a lot
      // of string new and delete calls.
      //
      // The temporary value is on the stack, so there will typically only be one
      // at a time.  But we can't guarantee that.  So here is a pool optimized for
      // the case where one element is allocated at any given time.  It can
      // allocate more, should the underlying C++ sort code change.
      {
        FreePool pool(element_size);
        // TODO is this necessary anymore?
  #if defined(_WIN32) || defined(_WIN64)
        std::stable_sort
  #else
        std::sort
#endif
          (SizedIterator(SizedProxy(start, pool)), SizedIterator(SizedProxy(end, pool)), SizedCompare<Compare>(compare));
    }
  }
}

} // namespace util

// Dirty hack because g++ 4.6 at least wants to do a bunch of copy operations.
namespace std {
inline void iter_swap(util::SizedIterator first, util::SizedIterator second) {
  util::swap(*first, *second);
}
} // namespace std
#endif // UTIL_SIZED_ITERATOR_H
