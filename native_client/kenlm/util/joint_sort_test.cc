#include "util/joint_sort.hh"

#define BOOST_TEST_MODULE JointSortTest
#include <boost/test/unit_test.hpp>

namespace util { namespace {

BOOST_AUTO_TEST_CASE(just_flip) {
  char keys[2];
  int values[2];
  keys[0] = 1; values[0] = 327;
  keys[1] = 0; values[1] = 87897;
  JointSort<char *, int *>(keys + 0, keys + 2, values + 0);
  BOOST_CHECK_EQUAL(0, keys[0]);
  BOOST_CHECK_EQUAL(87897, values[0]);
  BOOST_CHECK_EQUAL(1, keys[1]);
  BOOST_CHECK_EQUAL(327, values[1]);
}

BOOST_AUTO_TEST_CASE(three) {
  char keys[3];
  int values[3];
  keys[0] = 1; values[0] = 327;
  keys[1] = 2; values[1] = 87897;
  keys[2] = 0; values[2] = 10;
  JointSort<char *, int *>(keys + 0, keys + 3, values + 0);
  BOOST_CHECK_EQUAL(0, keys[0]);
  BOOST_CHECK_EQUAL(1, keys[1]);
  BOOST_CHECK_EQUAL(2, keys[2]);
}

BOOST_AUTO_TEST_CASE(char_int) {
  char keys[4];
  int values[4];
  keys[0] = 3; values[0] = 327;
  keys[1] = 1; values[1] = 87897;
  keys[2] = 2; values[2] = 10;
  keys[3] = 0; values[3] = 24347;
  JointSort<char *, int *>(keys + 0, keys + 4, values + 0);
  BOOST_CHECK_EQUAL(0, keys[0]);
  BOOST_CHECK_EQUAL(24347, values[0]);
  BOOST_CHECK_EQUAL(1, keys[1]);
  BOOST_CHECK_EQUAL(87897, values[1]);
  BOOST_CHECK_EQUAL(2, keys[2]);
  BOOST_CHECK_EQUAL(10, values[2]);
  BOOST_CHECK_EQUAL(3, keys[3]);
  BOOST_CHECK_EQUAL(327, values[3]);
}

BOOST_AUTO_TEST_CASE(swap_proxy) {
  char keys[2] = {0, 1};
  int values[2] = {2, 3};
  detail::JointProxy<char *, int *> first(keys, values);
  detail::JointProxy<char *, int *> second(keys + 1, values + 1);
  swap(first, second);
  BOOST_CHECK_EQUAL(1, keys[0]);
  BOOST_CHECK_EQUAL(0, keys[1]);
  BOOST_CHECK_EQUAL(3, values[0]);
  BOOST_CHECK_EQUAL(2, values[1]);
}

}} // namespace anonymous util
