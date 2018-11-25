#include "util/pcqueue.hh"

#define BOOST_TEST_MODULE PCQueueTest
#include <boost/test/unit_test.hpp>

namespace util {
namespace {

BOOST_AUTO_TEST_CASE(SingleThread) {
  PCQueue<int> queue(10);
  for (int i = 0; i < 10; ++i) {
    queue.Produce(i);
  }
  for (int i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(i, queue.Consume());
  }
}

}
} // namespace util
