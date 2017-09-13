#define BOOST_LEXICAL_CAST_ASSUME_C_LOCALE
#include "util/integer_to_string.hh"
#include "util/string_piece.hh"

#define BOOST_TEST_MODULE IntegerToStringTest
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>

#include <limits>

namespace util {
namespace {

template <class T> void TestValue(const T value) {
  char buf[ToStringBuf<T>::kBytes];
  StringPiece result(buf, ToString(value, buf) - buf);
  BOOST_REQUIRE_GE(static_cast<std::size_t>(ToStringBuf<T>::kBytes), result.size());
  if (value) {
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(value), result);
  } else {
    // Platforms can do void * as 0x0 or 0.
    BOOST_CHECK(result == "0x0" || result == "0");
  }
}

template <class T> void TestCorners() {
  TestValue(std::numeric_limits<T>::min());
  TestValue(std::numeric_limits<T>::max());
  TestValue((T)0);
  TestValue((T)-1);
  TestValue((T)1);
}

BOOST_AUTO_TEST_CASE(Corners) {
  TestCorners<uint16_t>();
  TestCorners<uint32_t>();
  TestCorners<uint64_t>();
  TestCorners<int16_t>();
  TestCorners<int32_t>();
  TestCorners<int64_t>();
  TestCorners<const void*>();
}

template <class T> void TestAll() {
  for (T i = std::numeric_limits<T>::min(); i < std::numeric_limits<T>::max(); ++i) {
    TestValue(i);
  }
  TestValue(std::numeric_limits<T>::max());
}

BOOST_AUTO_TEST_CASE(Short) {
  TestAll<uint16_t>();
  TestAll<int16_t>();
}

template <class T> void Test10s() {
  for (T i = 1; i < std::numeric_limits<T>::max() / 10; i *= 10) {
    TestValue(i);
    TestValue(i - 1);
    TestValue(i + 1);
  }
}

BOOST_AUTO_TEST_CASE(Tens) {
  Test10s<uint64_t>();
  Test10s<int64_t>();
  Test10s<uint32_t>();
  Test10s<int32_t>();
}

BOOST_AUTO_TEST_CASE(Pointers) {
  for (uintptr_t i = 1; i < std::numeric_limits<uintptr_t>::max() / 10; i *= 10) {
    TestValue((const void*)i);
  }
  for (uintptr_t i = 0; i < 256; ++i) {
    TestValue((const void*)i);
    TestValue((const void*)(i + 0xf00));
  }
}

}} // namespaces
