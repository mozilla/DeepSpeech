#define BOOST_LEXICAL_CAST_ASSUME_C_LOCALE
#define BOOST_TEST_MODULE FakeOStreamTest

#include "util/string_stream.hh"
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>

#include <cstddef>
#include <limits>

namespace util { namespace {

template <class T> void TestEqual(const T value) {
  StringStream strme;
  strme << value;
  BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(value), strme.str());
}

template <class T> void TestCorners() {
  TestEqual(std::numeric_limits<T>::max());
  TestEqual(std::numeric_limits<T>::min());
  TestEqual(static_cast<T>(0));
  TestEqual(static_cast<T>(-1));
  TestEqual(static_cast<T>(1));
}

BOOST_AUTO_TEST_CASE(Integer) {
  TestCorners<char>();
  TestCorners<signed char>();
  TestCorners<unsigned char>();

  TestCorners<short>();
  TestCorners<signed short>();
  TestCorners<unsigned short>();

  TestCorners<int>();
  TestCorners<unsigned int>();
  TestCorners<signed int>();

  TestCorners<long>();
  TestCorners<unsigned long>();
  TestCorners<signed long>();

  TestCorners<long long>();
  TestCorners<unsigned long long>();
  TestCorners<signed long long>();

  TestCorners<std::size_t>();
}

enum TinyEnum { EnumValue };

BOOST_AUTO_TEST_CASE(EnumCase) {
  TestEqual(EnumValue);
}

BOOST_AUTO_TEST_CASE(Strings) {
  TestEqual("foo");
  const char *a = "bar";
  TestEqual(a);
  StringPiece piece("abcdef");
  TestEqual(piece);
  TestEqual(StringPiece());

  char non_const[3];
  non_const[0] = 'b';
  non_const[1] = 'c';
  non_const[2] = 0;

  StringStream out;
  out << "a" << non_const << 'c';
  BOOST_CHECK_EQUAL("abcc", out.str());

  // Now test as a separate object.
  StringStream stream;
  stream << "a" << non_const << 'c' << piece;
  BOOST_CHECK_EQUAL("abccabcdef", stream.str());
}

}} // namespaces
