#include "util/sized_iterator.hh"

#define BOOST_TEST_MODULE SizedIteratorTest
#include <boost/test/unit_test.hpp>

namespace util { namespace {

struct CompareChar {
  bool operator()(const void *first, const void *second) const {
    return *static_cast<const char*>(first) < *static_cast<const char*>(second);
  }
};

BOOST_AUTO_TEST_CASE(sort) {
  char items[3] = {1, 2, 0};
  SizedSort(items, items + 3, 1, CompareChar());
  BOOST_CHECK_EQUAL(0, items[0]);
  BOOST_CHECK_EQUAL(1, items[1]);
  BOOST_CHECK_EQUAL(2, items[2]);
}

}} // namespace anonymous util
