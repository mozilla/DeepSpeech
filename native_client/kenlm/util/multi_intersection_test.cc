#include "util/multi_intersection.hh"

#define BOOST_TEST_MODULE MultiIntersectionTest
#include <boost/test/unit_test.hpp>

namespace util {
namespace {

BOOST_AUTO_TEST_CASE(Empty) {
  std::vector<boost::iterator_range<const unsigned int*> > sets;

  sets.push_back(boost::iterator_range<const unsigned int*>(static_cast<const unsigned int*>(NULL), static_cast<const unsigned int*>(NULL)));
  BOOST_CHECK(!FirstIntersection(sets));
}

BOOST_AUTO_TEST_CASE(Single) {
  std::vector<unsigned int> nums;
  nums.push_back(1);
  nums.push_back(4);
  nums.push_back(100);
  std::vector<boost::iterator_range<std::vector<unsigned int>::const_iterator> > sets;
  sets.push_back(nums);

  boost::optional<unsigned int> ret(FirstIntersection(sets));

  BOOST_REQUIRE(ret);
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(1), *ret);
}

template <class T, unsigned int len> boost::iterator_range<const T*> RangeFromArray(const T (&arr)[len]) {
  return boost::iterator_range<const T*>(arr, arr + len);
}

BOOST_AUTO_TEST_CASE(MultiNone) {
  unsigned int nums0[] = {1, 3, 4, 22};
  unsigned int nums1[] = {2, 5, 12};
  unsigned int nums2[] = {4, 17};

  std::vector<boost::iterator_range<const unsigned int*> > sets;
  sets.push_back(RangeFromArray(nums0));
  sets.push_back(RangeFromArray(nums1));
  sets.push_back(RangeFromArray(nums2));

  BOOST_CHECK(!FirstIntersection(sets));
}

BOOST_AUTO_TEST_CASE(MultiOne) {
  unsigned int nums0[] = {1, 3, 4, 17, 22};
  unsigned int nums1[] = {2, 5, 12, 17};
  unsigned int nums2[] = {4, 17};

  std::vector<boost::iterator_range<const unsigned int*> > sets;
  sets.push_back(RangeFromArray(nums0));
  sets.push_back(RangeFromArray(nums1));
  sets.push_back(RangeFromArray(nums2));

  boost::optional<unsigned int> ret(FirstIntersection(sets));
  BOOST_REQUIRE(ret);
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(17), *ret);
}

} // namespace
} // namespace util
