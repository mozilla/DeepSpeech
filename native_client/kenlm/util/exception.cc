#include "util/exception.hh"

#ifdef __GXX_RTTI
#include <typeinfo>
#endif

#include <cerrno>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <io.h>
#endif

namespace util {

Exception::Exception() throw() {}
Exception::~Exception() throw() {}

void Exception::SetLocation(const char *file, unsigned int line, const char *func, const char *child_name, const char *condition) {
  /* The child class might have set some text, but we want this to come first.
   * Another option would be passing this information to the constructor, but
   * then child classes would have to accept constructor arguments and pass
   * them down.
   */
  std::string old_text;
  what_.swap(old_text);
  what_ << file << ':' << line;
  if (func) what_ << " in " << func << " threw ";
  if (child_name) {
    what_ << child_name;
  } else {
#ifdef __GXX_RTTI
    what_ << typeid(this).name();
#else
    what_ << "an exception";
#endif
  }
  if (condition) {
    what_ << " because `" << condition << '\'';
  }
  what_ << ".\n";
  what_ << old_text;
}

namespace {

#ifdef __GNUC__
const char *HandleStrerror(int ret, const char *buf) __attribute__ ((unused));
const char *HandleStrerror(const char *ret, const char * /*buf*/) __attribute__ ((unused));
#endif
// At least one of these functions will not be called.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
// The XOPEN version.
const char *HandleStrerror(int ret, const char *buf) {
  if (!ret) return buf;
  return NULL;
}

// The GNU version.
const char *HandleStrerror(const char *ret, const char * /*buf*/) {
  return ret;
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace

ErrnoException::ErrnoException() throw() : errno_(errno) {
  char buf[200];
  buf[0] = 0;
#if defined(sun) || defined(_WIN32) || defined(_WIN64)
  const char *add = strerror(errno);
#else
  const char *add = HandleStrerror(strerror_r(errno, buf, 200), buf);
#endif

  if (add) {
    *this << add << ' ';
  }
}

ErrnoException::~ErrnoException() throw() {}

OverflowException::OverflowException() throw() {}
OverflowException::~OverflowException() throw() {}

#if defined(_WIN32) || defined(_WIN64)
WindowsException::WindowsException() throw() {
  unsigned int last_error = GetLastError();
  char error_msg[256] = "";
  if (!FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, last_error, LANG_NEUTRAL, error_msg, sizeof(error_msg), NULL)) {
    *this << "Windows error " << GetLastError() << " while formatting Windows error " << last_error << ". ";
  } else {
    *this << "Windows error " << last_error << ": " << error_msg;
  }
}
WindowsException::~WindowsException() throw() {}
#endif

} // namespace util
