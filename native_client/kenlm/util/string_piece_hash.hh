#ifndef UTIL_STRING_PIECE_HASH_H
#define UTIL_STRING_PIECE_HASH_H

#include "util/string_piece.hh"

#include <boost/functional/hash.hpp>
#include <boost/version.hpp>

inline size_t hash_value(const StringPiece &str) {
  return boost::hash_range(str.data(), str.data() + str.length());
}

/* Support for lookup of StringPiece in boost::unordered_map<std::string> */
struct StringPieceCompatibleHash : public std::unary_function<const StringPiece &, size_t> {
  size_t operator()(const StringPiece &str) const {
    return hash_value(str);
  }
};

struct StringPieceCompatibleEquals : public std::binary_function<const StringPiece &, const std::string &, bool> {
  bool operator()(const StringPiece &first, const StringPiece &second) const {
    return first == second;
  }
};
template <class T> typename T::const_iterator FindStringPiece(const T &t, const StringPiece &key) {
#if BOOST_VERSION < 104200
  std::string temp(key.data(), key.size());
  return t.find(temp);
#else
  return t.find(key, StringPieceCompatibleHash(), StringPieceCompatibleEquals());
#endif
}

template <class T> typename T::iterator FindStringPiece(T &t, const StringPiece &key) {
#if BOOST_VERSION < 104200
  std::string temp(key.data(), key.size());
  return t.find(temp);
#else
  return t.find(key, StringPieceCompatibleHash(), StringPieceCompatibleEquals());
#endif
}

#endif // UTIL_STRING_PIECE_HASH_H
