#include "lm/interpolate/normalize.hh"

#include "lm/interpolate/interpolate_info.hh"
#include "lm/interpolate/merge_probabilities.hh"
#include "lm/common/ngram_stream.hh"
#include "util/stream/chain.hh"
#include "util/stream/multi_stream.hh"

#define BOOST_TEST_MODULE NormalizeTest
#include <boost/test/unit_test.hpp>

namespace lm { namespace interpolate { namespace {

// log without backoff
const float kInputs[] = {-0.3, 1.2, -9.8, 4.0, -7.0, 0.0};

class WriteInput {
  public:
    WriteInput() {}
    void Run(const util::stream::ChainPosition &to) {
      util::stream::Stream out(to);
      for (WordIndex i = 0; i < sizeof(kInputs) / sizeof(float); ++i, ++out) {
        memcpy(out.Get(), &i, sizeof(WordIndex));
        memcpy((uint8_t*)out.Get() + sizeof(WordIndex), &kInputs[i], sizeof(float));
      }
      out.Poison();
    }
};

void CheckOutput(const util::stream::ChainPosition &from) {
  NGramStream<float> in(from);
  float sum = 0.0;
  for (WordIndex i = 0; i < sizeof(kInputs) / sizeof(float) - 1 /* <s> at the end */; ++i) {
    sum += pow(10.0, kInputs[i]);
  }
  sum = log10(sum);
  BOOST_REQUIRE(in);
  BOOST_CHECK_CLOSE(kInputs[0] - sum, in->Value(), 0.0001);
  BOOST_REQUIRE(++in);
  BOOST_CHECK_CLOSE(kInputs[1] - sum, in->Value(), 0.0001);
  BOOST_REQUIRE(++in);
  BOOST_CHECK_CLOSE(kInputs[2] - sum, in->Value(), 0.0001);
  BOOST_REQUIRE(++in);
  BOOST_CHECK_CLOSE(kInputs[3] - sum, in->Value(), 0.0001);
  BOOST_REQUIRE(++in);
  BOOST_CHECK_CLOSE(kInputs[4] - sum, in->Value(), 0.0001);
  BOOST_REQUIRE(++in);
  BOOST_CHECK_CLOSE(kInputs[5] - sum, in->Value(), 0.0001);
  BOOST_CHECK(!++in);
}

BOOST_AUTO_TEST_CASE(Unigrams) {
  InterpolateInfo info;
  info.lambdas.push_back(2.0);
  info.lambdas.push_back(-0.1);
  info.orders.push_back(1);
  info.orders.push_back(1);

  BOOST_CHECK_EQUAL(0, MakeEncoder(info, 1).EncodedLength());

  // No backoffs.
  util::stream::Chains blank(0);
  util::FixedArray<util::stream::ChainPositions> models_by_order(2);
  models_by_order.push_back(blank);
  models_by_order.push_back(blank);

  util::stream::Chains merged_probabilities(1);
  util::stream::Chains probabilities_out(1);
  util::stream::Chains backoffs_out(0);

  merged_probabilities.push_back(util::stream::ChainConfig(sizeof(WordIndex) + sizeof(float) + sizeof(float), 2, 24));
  probabilities_out.push_back(util::stream::ChainConfig(sizeof(WordIndex) + sizeof(float), 2, 100));

  merged_probabilities[0] >> WriteInput();
  Normalize(info, models_by_order, merged_probabilities, probabilities_out, backoffs_out);

  util::stream::ChainPosition checker(probabilities_out[0].Add());

  merged_probabilities >> util::stream::kRecycle;
  probabilities_out >> util::stream::kRecycle;

  CheckOutput(checker);
  probabilities_out.Wait();
}

}}} // namespaces
