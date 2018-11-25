#ifndef UTIL_SCOPED_H
#define UTIL_SCOPED_H
/* Other scoped objects in the style of scoped_ptr. */

#include "util/exception.hh"
#include <cstddef>
#include <cstdlib>

namespace util {

class MallocException : public ErrnoException {
  public:
    explicit MallocException(std::size_t requested) throw();
    ~MallocException() throw();
};

void *MallocOrThrow(std::size_t requested);
void *CallocOrThrow(std::size_t requested);

/* Unfortunately, defining the operator* for void * makes the compiler complain.
 * So scoped is specialized to void.  This includes the functionality common to
 * both, namely everything except reference.
 */
template <class T, class Closer> class scoped_base {
  public:
    explicit scoped_base(T *p = NULL) : p_(p) {}

    ~scoped_base() { Closer::Close(p_); }

#if __cplusplus >= 201103L
    scoped_base(scoped_base &&from) noexcept : p_(from.p_) {
      from.p_ = nullptr;
    }
#endif

    void reset(T *p = NULL) {
      scoped_base other(p_);
      p_ = p;
    }

    T *get() { return p_; }
    const T *get() const { return p_; }

    T *operator->() { return p_; }
    const T *operator->() const { return p_; }

    T *release() {
      T *ret = p_;
      p_ = NULL;
      return ret;
    }

  protected:
    T *p_;

#if __cplusplus >= 201103L
  public:
    scoped_base(const scoped_base &) = delete;
    scoped_base &operator=(const scoped_base &) = delete;
#else
  private:
    scoped_base(const scoped_base &);
    scoped_base &operator=(const scoped_base &);
#endif
};

template <class T, class Closer> class scoped : public scoped_base<T, Closer> {
  public:
    explicit scoped(T *p = NULL) : scoped_base<T, Closer>(p) {}

    T &operator*() { return *scoped_base<T, Closer>::p_; }
    const T&operator*() const { return *scoped_base<T, Closer>::p_; }
};

template <class Closer> class scoped<void, Closer> : public scoped_base<void, Closer> {
  public:
    explicit scoped(void *p = NULL) : scoped_base<void, Closer>(p) {}
};

/* Closer for c functions like std::free and cmph cleanup functions */
template <class T, void (*clean)(T*)> struct scoped_c_forward {
  static void Close(T *p) { clean(p); }
};
// Call a C function to delete stuff
template <class T, void (*clean)(T*)> class scoped_c : public scoped<T, scoped_c_forward<T, clean> > {
  public:
    explicit scoped_c(T *p = NULL) : scoped<T, scoped_c_forward<T, clean> >(p) {}
};

class scoped_malloc : public scoped_c<void, std::free> {
  public:
    explicit scoped_malloc(void *p = NULL) : scoped_c<void, std::free>(p) {}

    explicit scoped_malloc(std::size_t size) : scoped_c<void, std::free>(MallocOrThrow(size)) {}

    void call_realloc(std::size_t to);
};

/* scoped_array using delete[] */
struct scoped_delete_array_forward {
  template <class T> static void Close(T *p) { delete [] p; }
};
// Hat tip to boost.
template <class T> class scoped_array : public scoped<T, scoped_delete_array_forward> {
  public:
    explicit scoped_array(T *p = NULL) : scoped<T, scoped_delete_array_forward>(p) {}

    T &operator[](std::size_t idx) { return scoped<T, scoped_delete_array_forward>::p_[idx]; }
    const T &operator[](std::size_t idx) const { return scoped<T, scoped_delete_array_forward>::p_[idx]; }
};

/* scoped_ptr using delete.  If only there were a template typedef. */
struct scoped_delete_forward {
  template <class T> static void Close(T *p) { delete p; }
};
template <class T> class scoped_ptr : public scoped<T, scoped_delete_forward> {
  public:
    explicit scoped_ptr(T *p = NULL) : scoped<T, scoped_delete_forward>(p) {}
};

void AdviseHugePages(const void *addr, std::size_t size);

} // namespace util

#endif // UTIL_SCOPED_H
