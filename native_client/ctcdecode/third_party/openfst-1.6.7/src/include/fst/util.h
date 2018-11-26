// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// FST utility inline definitions.

#ifndef FST_UTIL_H_
#define FST_UTIL_H_

#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fst/compat.h>
#include <fst/types.h>
#include <fst/log.h>
#include <fstream>

#include <fst/flags.h>


// Utility for error handling.

DECLARE_bool(fst_error_fatal);

#define FSTERROR() \
  (FLAGS_fst_error_fatal ? LOG(FATAL) : LOG(ERROR))

namespace fst {

// Utility for type I/O.

// Reads types from an input stream.

// Generic case.
template <class T,
    typename std::enable_if<std::is_class<T>::value, T>::type* = nullptr>
inline std::istream &ReadType(std::istream &strm, T *t) {
  return t->Read(strm);
}

// Numeric (boolean, integral, floating-point) case.
template <class T,
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type* = nullptr>
inline std::istream &ReadType(std::istream &strm, T *t) {
  return strm.read(reinterpret_cast<char *>(t), sizeof(T)); \
}

// String case.
inline std::istream &ReadType(std::istream &strm, string *s) {  // NOLINT
  s->clear();
  int32 ns = 0;
  strm.read(reinterpret_cast<char *>(&ns), sizeof(ns));
  for (int32 i = 0; i < ns; ++i) {
    char c;
    strm.read(&c, 1);
    *s += c;
  }
  return strm;
}

// Declares types that can be read from an input stream.
template <class... T>
std::istream &ReadType(std::istream &strm, std::vector<T...> *c);
template <class... T>
std::istream &ReadType(std::istream &strm, std::list<T...> *c);
template <class... T>
std::istream &ReadType(std::istream &strm, std::set<T...> *c);
template <class... T>
std::istream &ReadType(std::istream &strm, std::map<T...> *c);
template <class... T>
std::istream &ReadType(std::istream &strm, std::unordered_map<T...> *c);
template <class... T>
std::istream &ReadType(std::istream &strm, std::unordered_set<T...> *c);

// Pair case.
template <typename S, typename T>
inline std::istream &ReadType(std::istream &strm, std::pair<S, T> *p) {
  ReadType(strm, &p->first);
  ReadType(strm, &p->second);
  return strm;
}

template <typename S, typename T>
inline std::istream &ReadType(std::istream &strm, std::pair<const S, T> *p) {
  ReadType(strm, const_cast<S *>(&p->first));
  ReadType(strm, &p->second);
  return strm;
}

namespace internal {
template <class C, class ReserveFn>
std::istream &ReadContainerType(std::istream &strm, C *c, ReserveFn reserve) {
  c->clear();
  int64 n = 0;
  ReadType(strm, &n);
  reserve(c, n);
  auto insert = std::inserter(*c, c->begin());
  for (int64 i = 0; i < n; ++i) {
    typename C::value_type value;
    ReadType(strm, &value);
    *insert = value;
  }
  return strm;
}
}  // namespace internal

template <class... T>
std::istream &ReadType(std::istream &strm, std::vector<T...> *c) {
  return internal::ReadContainerType(
      strm, c, [](decltype(c) v, int n) { v->reserve(n); });
}

template <class... T>
std::istream &ReadType(std::istream &strm, std::list<T...> *c) {
  return internal::ReadContainerType(strm, c, [](decltype(c) v, int n) {});
}

template <class... T>
std::istream &ReadType(std::istream &strm, std::set<T...> *c) {
  return internal::ReadContainerType(strm, c, [](decltype(c) v, int n) {});
}

template <class... T>
std::istream &ReadType(std::istream &strm, std::map<T...> *c) {
  return internal::ReadContainerType(strm, c, [](decltype(c) v, int n) {});
}

template <class... T>
std::istream &ReadType(std::istream &strm, std::unordered_set<T...> *c) {
  return internal::ReadContainerType(
      strm, c, [](decltype(c) v, int n) { v->reserve(n); });
}

template <class... T>
std::istream &ReadType(std::istream &strm, std::unordered_map<T...> *c) {
  return internal::ReadContainerType(
      strm, c, [](decltype(c) v, int n) { v->reserve(n); });
}

// Writes types to an output stream.

// Generic case.
template <class T,
    typename std::enable_if<std::is_class<T>::value, T>::type* = nullptr>
inline std::ostream &WriteType(std::ostream &strm, const T t) {
  t.Write(strm);
  return strm;
}

// Numeric (boolean, integral, floating-point) case.
template <class T,
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type* = nullptr>
inline std::ostream &WriteType(std::ostream &strm, const T t) {
  return strm.write(reinterpret_cast<const char *>(&t), sizeof(T));
}

// String case.
inline std::ostream &WriteType(std::ostream &strm, const string &s) {  // NOLINT
  int32 ns = s.size();
  strm.write(reinterpret_cast<const char *>(&ns), sizeof(ns));
  return strm.write(s.data(), ns);
}

// Declares types that can be written to an output stream.

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::vector<T...> &c);
template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::list<T...> &c);
template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::set<T...> &c);
template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::map<T...> &c);
template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::unordered_map<T...> &c);
template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::unordered_set<T...> &c);

// Pair case.
template <typename S, typename T>
inline std::ostream &WriteType(std::ostream &strm,
                               const std::pair<S, T> &p) {  // NOLINT
  WriteType(strm, p.first);
  WriteType(strm, p.second);
  return strm;
}

namespace internal {
template <class C>
std::ostream &WriteContainer(std::ostream &strm, const C &c) {
  const int64 n = c.size();
  WriteType(strm, n);
  for (const auto &e : c) {
    WriteType(strm, e);
  }
  return strm;
}
}  // namespace internal

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::vector<T...> &c) {
  return internal::WriteContainer(strm, c);
}

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::list<T...> &c) {
  return internal::WriteContainer(strm, c);
}

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::set<T...> &c) {
  return internal::WriteContainer(strm, c);
}

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::map<T...> &c) {
  return internal::WriteContainer(strm, c);
}

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::unordered_map<T...> &c) {
  return internal::WriteContainer(strm, c);
}

template <typename... T>
std::ostream &WriteType(std::ostream &strm, const std::unordered_set<T...> &c) {
  return internal::WriteContainer(strm, c);
}

// Utilities for converting between int64 or Weight and string.

int64 StrToInt64(const string &s, const string &src, size_t nline,
                 bool allow_negative, bool *error = nullptr);

template <typename Weight>
Weight StrToWeight(const string &s, const string &src, size_t nline) {
  Weight w;
  std::istringstream strm(s);
  strm >> w;
  if (!strm) {
    FSTERROR() << "StrToWeight: Bad weight = \"" << s << "\", source = " << src
               << ", line = " << nline;
    return Weight::NoWeight();
  }
  return w;
}

template <typename Weight>
void WeightToStr(Weight w, string *s) {
  std::ostringstream strm;
  strm.precision(9);
  strm << w;
  s->append(strm.str().data(), strm.str().size());
}

// Utilities for reading/writing integer pairs (typically labels)

// Modifies line using a vector of pointers to a buffer beginning with line.
void SplitString(char *line, const char *delim, std::vector<char *> *vec,
                 bool omit_empty_strings);

template <typename I>
bool ReadIntPairs(const string &filename, std::vector<std::pair<I, I>> *pairs,
                  bool allow_negative = false) {
  std::ifstream strm(filename, std::ios_base::in);
  if (!strm) {
    LOG(ERROR) << "ReadIntPairs: Can't open file: " << filename;
    return false;
  }
  const int kLineLen = 8096;
  char line[kLineLen];
  size_t nline = 0;
  pairs->clear();
  while (strm.getline(line, kLineLen)) {
    ++nline;
    std::vector<char *> col;
    SplitString(line, "\n\t ", &col, true);
    // empty line or comment?
    if (col.empty() || col[0][0] == '\0' || col[0][0] == '#') continue;
    if (col.size() != 2) {
      LOG(ERROR) << "ReadIntPairs: Bad number of columns, "
                 << "file = " << filename << ", line = " << nline;
      return false;
    }
    bool err;
    I i1 = StrToInt64(col[0], filename, nline, allow_negative, &err);
    if (err) return false;
    I i2 = StrToInt64(col[1], filename, nline, allow_negative, &err);
    if (err) return false;
    pairs->push_back(std::make_pair(i1, i2));
  }
  return true;
}

template <typename I>
bool WriteIntPairs(const string &filename,
                   const std::vector<std::pair<I, I>> &pairs) {
  std::ostream *strm = &std::cout;
  if (!filename.empty()) {
    strm = new std::ofstream(filename);
    if (!*strm) {
      LOG(ERROR) << "WriteIntPairs: Can't open file: " << filename;
      return false;
    }
  }
  for (ssize_t n = 0; n < pairs.size(); ++n) {
    *strm << pairs[n].first << "\t" << pairs[n].second << "\n";
  }
  if (!*strm) {
    LOG(ERROR) << "WriteIntPairs: Write failed: "
               << (filename.empty() ? "standard output" : filename);
    return false;
  }
  if (strm != &std::cout) delete strm;
  return true;
}

// Utilities for reading/writing label pairs.

template <typename Label>
bool ReadLabelPairs(const string &filename,
                    std::vector<std::pair<Label, Label>> *pairs,
                    bool allow_negative = false) {
  return ReadIntPairs(filename, pairs, allow_negative);
}

template <typename Label>
bool WriteLabelPairs(const string &filename,
                     const std::vector<std::pair<Label, Label>> &pairs) {
  return WriteIntPairs(filename, pairs);
}

// Utilities for converting a type name to a legal C symbol.

void ConvertToLegalCSymbol(string *s);

// Utilities for stream I/O.

bool AlignInput(std::istream &strm);
bool AlignOutput(std::ostream &strm);

// An associative container for which testing membership is faster than an STL
// set if members are restricted to an interval that excludes most non-members.
// A Key must have ==, !=, and < operators defined. Element NoKey should be a
// key that marks an uninitialized key and is otherwise unused. Find() returns
// an STL const_iterator to the match found, otherwise it equals End().
template <class Key, Key NoKey>
class CompactSet {
 public:
  using const_iterator = typename std::set<Key>::const_iterator;

  CompactSet() : min_key_(NoKey), max_key_(NoKey) {}

  CompactSet(const CompactSet<Key, NoKey> &compact_set)
      : set_(compact_set.set_),
        min_key_(compact_set.min_key_),
        max_key_(compact_set.max_key_) {}

  void Insert(Key key) {
    set_.insert(key);
    if (min_key_ == NoKey || key < min_key_) min_key_ = key;
    if (max_key_ == NoKey || max_key_ < key) max_key_ = key;
  }

  void Erase(Key key) {
    set_.erase(key);
    if (set_.empty()) {
      min_key_ = max_key_ = NoKey;
    } else if (key == min_key_) {
      ++min_key_;
    } else if (key == max_key_) {
      --max_key_;
    }
  }

  void Clear() {
    set_.clear();
    min_key_ = max_key_ = NoKey;
  }

  const_iterator Find(Key key) const {
    if (min_key_ == NoKey || key < min_key_ || max_key_ < key) {
      return set_.end();
    } else {
      return set_.find(key);
    }
  }

  bool Member(Key key) const {
    if (min_key_ == NoKey || key < min_key_ || max_key_ < key) {
      return false;  // out of range
    } else if (min_key_ != NoKey && max_key_ + 1 == min_key_ + set_.size()) {
      return true;  // dense range
    } else {
      return set_.count(key);
    }
  }

  const_iterator Begin() const { return set_.begin(); }

  const_iterator End() const { return set_.end(); }

  // All stored keys are greater than or equal to this value.
  Key LowerBound() const { return min_key_; }

  // All stored keys are less than or equal to this value.
  Key UpperBound() const { return max_key_; }

 private:
  std::set<Key> set_;
  Key min_key_;
  Key max_key_;

  void operator=(const CompactSet &) = delete;
};

}  // namespace fst

#endif  // FST_UTIL_H_
