/* If you use ICU in your program, then compile with -DHAVE_ICU -licui18n.  If
 * you don't use ICU, then this will use the Google implementation from Chrome.
 * This has been modified from the original version to let you choose.
 */

// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// Copied from strings/stringpiece.h with modifications
//
// A string-like object that points to a sized piece of memory.
//
// Functions or methods may use const StringPiece& parameters to accept either
// a "const char*" or a "string" value that will be implicitly converted to
// a StringPiece.  The implicit conversion means that it is often appropriate
// to include this .h file in other files rather than forward-declaring
// StringPiece as would be appropriate for most other Google classes.
//
// Systematic usage of StringPiece is encouraged as it will reduce unnecessary
// conversions from "const char*" to "string" and back again.
//

#ifndef UTIL_STRING_PIECE_H
#define UTIL_STRING_PIECE_H

#include "util/have.hh"

#include <cstring>
#include <iosfwd>
#include <ostream>

#ifdef HAVE_ICU
#include <unicode/stringpiece.h>
#include <unicode/uversion.h>

// Old versions of ICU don't define operator== and operator!=.
#if (U_ICU_VERSION_MAJOR_NUM < 4) || ((U_ICU_VERSION_MAJOR_NUM == 4) && (U_ICU_VERSION_MINOR_NUM < 4))
#warning You are using an old version of ICU.  Consider upgrading to ICU >= 4.6.
inline bool operator==(const StringPiece& x, const StringPiece& y) {
  if (x.size() != y.size())
    return false;

  return std::memcmp(x.data(), y.data(), x.size()) == 0;
}

inline bool operator!=(const StringPiece& x, const StringPiece& y) {
  return !(x == y);
}
#endif // old version of ICU

U_NAMESPACE_BEGIN

inline bool starts_with(const StringPiece& longer, const StringPiece& prefix) {
  int longersize = longer.size(), prefixsize = prefix.size();
  return longersize >= prefixsize && std::memcmp(longer.data(), prefix.data(), prefixsize) == 0;
}

#else

#include <algorithm>
#include <cstddef>
#include <string>
#include <cstring>

#ifdef WIN32
#undef max
#undef min
#endif

class StringPiece {
 public:
  typedef size_t size_type;

 private:
  const char*   ptr_;
  size_type     length_;

 public:
  // We provide non-explicit singleton constructors so users can pass
  // in a "const char*" or a "string" wherever a "StringPiece" is
  // expected.
  StringPiece() : ptr_(NULL), length_(0) { }
  StringPiece(const char* str)
    : ptr_(str), length_((str == NULL) ? 0 : strlen(str)) { }
  StringPiece(const std::string& str)
    : ptr_(str.data()), length_(str.size()) { }
  StringPiece(const char* offset, size_type len)
    : ptr_(offset), length_(len) { }

  // data() may return a pointer to a buffer with embedded NULs, and the
  // returned buffer may or may not be null terminated.  Therefore it is
  // typically a mistake to pass data() to a routine that expects a NUL
  // terminated string.
  const char* data() const { return ptr_; }
  size_type size() const { return length_; }
  size_type length() const { return length_; }
  bool empty() const { return length_ == 0; }

  void clear() { ptr_ = NULL; length_ = 0; }
  void set(const char* data, size_type len) { ptr_ = data; length_ = len; }
  void set(const char* str) {
    ptr_ = str;
    length_ = str ? strlen(str) : 0;
  }
  void set(const void* data, size_type len) {
    ptr_ = reinterpret_cast<const char*>(data);
    length_ = len;
  }

  char operator[](size_type i) const { return ptr_[i]; }

  void remove_prefix(size_type n) {
    ptr_ += n;
    length_ -= n;
  }

  void remove_suffix(size_type n) {
    length_ -= n;
  }

  int compare(const StringPiece& x) const {
    int r = wordmemcmp(ptr_, x.ptr_, std::min(length_, x.length_));
    if (r == 0) {
      if (length_ < x.length_) r = -1;
      else if (length_ > x.length_) r = +1;
    }
    return r;
  }

  std::string as_string() const {
    // std::string doesn't like to take a NULL pointer even with a 0 size.
    return std::string(!empty() ? data() : "", size());
  }

  void CopyToString(std::string* target) const;
  void AppendToString(std::string* target) const;

  // Does "this" start with "x"
  bool starts_with(const StringPiece& x) const {
    return ((length_ >= x.length_) &&
            (wordmemcmp(ptr_, x.ptr_, x.length_) == 0));
  }

  // Does "this" end with "x"
  bool ends_with(const StringPiece& x) const {
    return ((length_ >= x.length_) &&
            (wordmemcmp(ptr_ + (length_-x.length_), x.ptr_, x.length_) == 0));
  }

  // standard STL container boilerplate
  typedef char value_type;
  typedef const char* pointer;
  typedef const char& reference;
  typedef const char& const_reference;
  typedef ptrdiff_t difference_type;
  static const size_type npos;
  typedef const char* const_iterator;
  typedef const char* iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  iterator begin() const { return ptr_; }
  iterator end() const { return ptr_ + length_; }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(ptr_ + length_);
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(ptr_);
  }

  size_type max_size() const { return length_; }
  size_type capacity() const { return length_; }

  size_type copy(char* buf, size_type n, size_type pos = 0) const;

  size_type find(const StringPiece& s, size_type pos = 0) const;
  size_type find(char c, size_type pos = 0) const;
  size_type rfind(const StringPiece& s, size_type pos = npos) const;
  size_type rfind(char c, size_type pos = npos) const;

  size_type find_first_of(const StringPiece& s, size_type pos = 0) const;
  size_type find_first_of(char c, size_type pos = 0) const {
    return find(c, pos);
  }
  size_type find_first_not_of(const StringPiece& s, size_type pos = 0) const;
  size_type find_first_not_of(char c, size_type pos = 0) const;
  size_type find_last_of(const StringPiece& s, size_type pos = npos) const;
  size_type find_last_of(char c, size_type pos = npos) const {
    return rfind(c, pos);
  }
  size_type find_last_not_of(const StringPiece& s, size_type pos = npos) const;
  size_type find_last_not_of(char c, size_type pos = npos) const;

  StringPiece substr(size_type pos, size_type n = npos) const;

  static int wordmemcmp(const char* p, const char* p2, size_type N) {
    return std::memcmp(p, p2, N);
  }
};

inline bool operator==(const StringPiece& x, const StringPiece& y) {
  if (x.size() != y.size())
    return false;

  return std::memcmp(x.data(), y.data(), x.size()) == 0;
}

inline bool operator!=(const StringPiece& x, const StringPiece& y) {
  return !(x == y);
}

inline bool starts_with(const StringPiece& longer, const StringPiece& prefix) {
  return longer.starts_with(prefix);
}

#endif // HAVE_ICU undefined

inline bool operator<(const StringPiece& x, const StringPiece& y) {
  const int r = std::memcmp(x.data(), y.data(),
                                       std::min(x.size(), y.size()));
  return ((r < 0) || ((r == 0) && (x.size() < y.size())));
}

inline bool operator>(const StringPiece& x, const StringPiece& y) {
  return y < x;
}

inline bool operator<=(const StringPiece& x, const StringPiece& y) {
  return !(x > y);
}

inline bool operator>=(const StringPiece& x, const StringPiece& y) {
  return !(x < y);
}

// allow StringPiece to be logged (needed for unit testing).
inline std::ostream& operator<<(std::ostream& o, const StringPiece& piece) {
  return o.write(piece.data(), static_cast<std::streamsize>(piece.size()));
}

#ifdef HAVE_ICU
U_NAMESPACE_END
using U_NAMESPACE_QUALIFIER StringPiece;
#endif

#endif  // UTIL_STRING_PIECE_H
