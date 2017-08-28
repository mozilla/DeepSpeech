#ifndef UTIL_MULTI_INTERSECTION_H
#define UTIL_MULTI_INTERSECTION_H

#include <boost/optional.hpp>
#include <boost/range/iterator_range.hpp>

#include <algorithm>
#include <functional>
#include <vector>

namespace util {

namespace detail {
template <class Range> struct RangeLessBySize : public std::binary_function<const Range &, const Range &, bool> {
  bool operator()(const Range &left, const Range &right) const {
    return left.size() < right.size();
  }
};

/* Takes sets specified by their iterators and a boost::optional containing
 * the lowest intersection if any.  Each set must be sorted in increasing
 * order.  sets is changed to truncate the beginning of each sequence to the
 * location of the match or an empty set.  Precondition: sets is not empty
 * since the intersection over null is the universe and this function does not
 * know the universe.
 */
template <class Iterator, class Less> boost::optional<typename std::iterator_traits<Iterator>::value_type> FirstIntersectionSorted(std::vector<boost::iterator_range<Iterator> > &sets, const Less &less = std::less<typename std::iterator_traits<Iterator>::value_type>()) {
  typedef std::vector<boost::iterator_range<Iterator> > Sets;
  typedef typename std::iterator_traits<Iterator>::value_type Value;

  assert(!sets.empty());

  if (sets.front().empty()) return boost::optional<Value>();
  // Possibly suboptimal to copy for general Value; makes unsigned int go slightly faster.
  Value highest(sets.front().front());
  for (typename Sets::iterator i(sets.begin()); i != sets.end(); ) {
    i->advance_begin(std::lower_bound(i->begin(), i->end(), highest, less) - i->begin());
    if (i->empty()) return boost::optional<Value>();
    if (less(highest, i->front())) {
      highest = i->front();
      // start over
      i = sets.begin();
    } else {
      ++i;
    }
  }
  return boost::optional<Value>(highest);
}

} // namespace detail

template <class Iterator, class Less> boost::optional<typename std::iterator_traits<Iterator>::value_type> FirstIntersection(std::vector<boost::iterator_range<Iterator> > &sets, const Less less) {
  assert(!sets.empty());

  std::sort(sets.begin(), sets.end(), detail::RangeLessBySize<boost::iterator_range<Iterator> >());
  return detail::FirstIntersectionSorted(sets, less);
}

template <class Iterator> boost::optional<typename std::iterator_traits<Iterator>::value_type> FirstIntersection(std::vector<boost::iterator_range<Iterator> > &sets) {
  return FirstIntersection(sets, std::less<typename std::iterator_traits<Iterator>::value_type>());
}

template <class Iterator, class Output, class Less> void AllIntersection(std::vector<boost::iterator_range<Iterator> > &sets, Output &out, const Less less) {
  typedef typename std::iterator_traits<Iterator>::value_type Value;
  assert(!sets.empty());

  std::sort(sets.begin(), sets.end(), detail::RangeLessBySize<boost::iterator_range<Iterator> >());
  boost::optional<Value> ret;
  for (boost::optional<Value> ret; (ret = detail::FirstIntersectionSorted(sets, less)); sets.front().advance_begin(1)) {
    out(*ret);
  }
}

template <class Iterator, class Output> void AllIntersection(std::vector<boost::iterator_range<Iterator> > &sets, Output &out) {
  AllIntersection(sets, out, std::less<typename std::iterator_traits<Iterator>::value_type>());
}

} // namespace util

#endif // UTIL_MULTI_INTERSECTION_H
