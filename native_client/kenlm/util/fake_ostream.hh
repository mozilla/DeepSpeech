#ifndef UTIL_FAKE_OSTREAM_H
#define UTIL_FAKE_OSTREAM_H

#include "util/float_to_string.hh"
#include "util/integer_to_string.hh"
#include "util/string_piece.hh"

#include <cassert>
#include <limits>

#include <stdint.h>

namespace util {

/* Like std::ostream but without being incredibly slow.
 * Supports most of the built-in types except for long double.
 *
 * The FakeOStream class is intended to be inherited from.  The inherting class
 * should provide:
 * public:
 *   Derived &flush();
 *   Derived &write(const void *data, std::size_t length);
 *
 * private: or protected:
 *   friend class FakeOStream;
 *   char *Ensure(std::size_t amount);
 *   void AdvanceTo(char *to);
 *
 * The Ensure function makes enough space for an in-place write and returns
 * where to write.  The AdvanceTo function happens after the write, saying how
 * much was actually written.
 *
 * Precondition:
 * amount <= kToStringMaxBytes for in-place writes.
 */
template <class Derived> class FakeOStream {
  public:
    FakeOStream() {}

    // This also covers std::string and char*
    Derived &operator<<(StringPiece str) {
      return C().write(str.data(), str.size());
    }

    // Handle integers by size and signedness.
  private:
    template <class Arg> struct EnableIfKludge {
      typedef Derived type;
    };
    template <class From, unsigned Length = sizeof(From), bool Signed = std::numeric_limits<From>::is_signed, bool IsInteger = std::numeric_limits<From>::is_integer> struct Coerce {};

    template <class From> struct Coerce<From, 2, false, true> { typedef uint16_t To; };
    template <class From> struct Coerce<From, 4, false, true> { typedef uint32_t To; };
    template <class From> struct Coerce<From, 8, false, true> { typedef uint64_t To; };

    template <class From> struct Coerce<From, 2, true, true> { typedef int16_t To; };
    template <class From> struct Coerce<From, 4, true, true> { typedef int32_t To; };
    template <class From> struct Coerce<From, 8, true, true> { typedef int64_t To; };
  public:
    template <class From> typename EnableIfKludge<typename Coerce<From>::To>::type &operator<<(const From value) {
      return CallToString(static_cast<typename Coerce<From>::To>(value));
    }

    // Character types that get copied as bytes instead of displayed as integers.
    Derived &operator<<(char val) { return put(val); }
    Derived &operator<<(signed char val) { return put(static_cast<char>(val)); }
    Derived &operator<<(unsigned char val) { return put(static_cast<char>(val)); }

    Derived &operator<<(bool val) { return put(val + '0'); }
    // enums will fall back to int but are not caught by the template.
    Derived &operator<<(int val) { return CallToString(static_cast<typename Coerce<int>::To>(val)); }

    Derived &operator<<(float val) { return CallToString(val); }
    Derived &operator<<(double val) { return CallToString(val); }

    // This is here to catch all the other pointer types.
    Derived &operator<<(const void *value) { return CallToString(value); }
    // This is here because the above line also catches const char*.
    Derived &operator<<(const char *value) { return *this << StringPiece(value); }
    Derived &operator<<(char *value) { return *this << StringPiece(value); }

    Derived &put(char val) {
      char *c = C().Ensure(1);
      *c = val;
      C().AdvanceTo(++c);
      return C();
    }

    char widen(char val) const { return val; }

  private:
    // References to derived class for convenience.
    Derived &C() {
      return *static_cast<Derived*>(this);
    }

    const Derived &C() const {
      return *static_cast<const Derived*>(this);
    }

    // This is separate to prevent an infinite loop if the compiler considers
    // types the same (i.e. gcc std::size_t and uint64_t or uint32_t).
    template <class T> Derived &CallToString(const T value) {
      C().AdvanceTo(ToString(value, C().Ensure(ToStringBuf<T>::kBytes)));
      return C();
    }
};

} // namespace

#endif // UTIL_FAKE_OSTREAM_H
