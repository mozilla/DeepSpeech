#include "lm/common/model_buffer.hh"
#include "lm/model.hh"
#include "lm/state.hh"

#define BOOST_TEST_MODULE ModelBufferTest
#include <boost/test/unit_test.hpp>

namespace lm { namespace {

BOOST_AUTO_TEST_CASE(Query) {
  std::string dir("test_data/");
  if (boost::unit_test::framework::master_test_suite().argc == 2) {
    dir = boost::unit_test::framework::master_test_suite().argv[1];
  }
  ngram::Model ref((dir + "/toy0.arpa").c_str());
  ModelBuffer test(dir + "/toy0");
  ngram::State ref_state, test_state;
  WordIndex a = ref.GetVocabulary().Index("a");
  BOOST_CHECK_CLOSE(
      ref.FullScore(ref.BeginSentenceState(), a, ref_state).prob,
      test.SlowQuery(ref.BeginSentenceState(), a, test_state),
      0.001);
  BOOST_CHECK_EQUAL((unsigned)ref_state.length, (unsigned)test_state.length);
  BOOST_CHECK_EQUAL(ref_state.words[0], test_state.words[0]);
  BOOST_CHECK_EQUAL(ref_state.backoff[0], test_state.backoff[0]);
  BOOST_CHECK(ref_state == test_state);

  ngram::State ref_state2, test_state2;
  WordIndex b = ref.GetVocabulary().Index("b");
  BOOST_CHECK_CLOSE(
      ref.FullScore(ref_state, b, ref_state2).prob,
      test.SlowQuery(test_state, b, test_state2),
      0.001);
  BOOST_CHECK(ref_state2 == test_state2);
  BOOST_CHECK_EQUAL(ref_state2.backoff[0], test_state2.backoff[0]);

  BOOST_CHECK_CLOSE(
      ref.FullScore(ref_state2, 0, ref_state).prob,
      test.SlowQuery(test_state2, 0, test_state),
      0.001);
  // The reference does state minimization but this doesn't.
}

}} // namespaces
