#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"

#define BOOST_TEST_MODULE TokenIteratorTest
#include <boost/test/unit_test.hpp>

#include <iostream>

namespace util {
namespace {

BOOST_AUTO_TEST_CASE(pipe_pipe_none) {
  const char str[] = "nodelimit at all";
  TokenIter<MultiCharacter> it(str, MultiCharacter("|||"));
  BOOST_REQUIRE(it);
  BOOST_CHECK_EQUAL(StringPiece(str), *it);
  ++it;
  BOOST_CHECK(!it);
}
BOOST_AUTO_TEST_CASE(pipe_pipe_two) {
  const char str[] = "|||";
  TokenIter<MultiCharacter> it(str, MultiCharacter("|||"));
  BOOST_REQUIRE(it);
  BOOST_CHECK_EQUAL(StringPiece(), *it);
  ++it;
  BOOST_REQUIRE(it);
  BOOST_CHECK_EQUAL(StringPiece(), *it);
  ++it;
  BOOST_CHECK(!it);
}

BOOST_AUTO_TEST_CASE(remove_empty) {
  const char str[] = "|||";
  TokenIter<MultiCharacter, true> it(str, MultiCharacter("|||"));
  BOOST_CHECK(!it);
}

BOOST_AUTO_TEST_CASE(remove_empty_keep) {
  const char str[] = " |||";
  TokenIter<MultiCharacter, true> it(str, MultiCharacter("|||"));
  BOOST_REQUIRE(it);
  BOOST_CHECK_EQUAL(StringPiece(" "), *it);
  ++it;
  BOOST_CHECK(!it);
}

} // namespace
} // namespace util
