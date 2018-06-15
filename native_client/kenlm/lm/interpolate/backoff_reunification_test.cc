#include "lm/interpolate/backoff_reunification.hh"
#include "lm/common/ngram_stream.hh"

#define BOOST_TEST_MODULE InterpolateBackoffReunificationTest
#include <boost/test/unit_test.hpp>

namespace lm {
namespace interpolate {

namespace {

// none of this input actually makes sense, all we care about is making
// sure the merging works
template <uint8_t N>
struct Gram {
  WordIndex ids[N];
  float prob;
  float boff;
};

template <uint8_t N>
struct Grams {
  const static Gram<N> grams[];
};

template <>
const Gram<1> Grams<1>::grams[]
    = {{{0}, -0.1f, -0.1f}, {{1}, -0.4f, -0.2f}, {{2}, -0.5f, -0.1f}};

template <>
const Gram<2> Grams<2>::grams[] = {{{0, 0}, -0.05f, -0.05f},
                                   {{1, 0}, -0.05f, -0.02f},
                                   {{1, 1}, -0.2f, -0.04f},
                                   {{2, 2}, -0.2f, -0.01f}};

template <>
const Gram<3> Grams<3>::grams[] = {{{0, 0, 0}, -0.001f, -0.005f},
                                   {{1, 0, 0}, -0.001f, -0.002f},
                                   {{2, 0, 0}, -0.001f, -0.003f},
                                   {{0, 1, 0}, -0.1f, -0.008f},
                                   {{1, 1, 0}, -0.1f, -0.09f},
                                   {{1, 1, 1}, -0.2f, -0.08f}};

template <uint8_t N>
class WriteInput {
public:
  void Run(const util::stream::ChainPosition &position) {
    lm::NGramStream<float> output(position);

    for (std::size_t i = 0; i < sizeof(Grams<N>::grams) / sizeof(Gram<N>);
         ++i, ++output) {
      std::copy(Grams<N>::grams[i].ids, Grams<N>::grams[i].ids + N,
                output->begin());
      output->Value() = Grams<N>::grams[i].prob;
    }
    output.Poison();
  }
};

template <uint8_t N>
class WriteBackoffs {
public:
  void Run(const util::stream::ChainPosition &position) {
    util::stream::Stream output(position);

    for (std::size_t i = 0; i < sizeof(Grams<N>::grams) / sizeof(Gram<N>);
         ++i, ++output) {
      *reinterpret_cast<float *>(output.Get()) = Grams<N>::grams[i].boff;
    }
    output.Poison();
  }
};

template <uint8_t N>
class CheckOutput {
public:
  void Run(const util::stream::ChainPosition &position) {
    lm::NGramStream<ProbBackoff> stream(position);

    std::size_t i = 0;
    for (; stream; ++stream, ++i) {
      std::stringstream ss;
      for (WordIndex *idx = stream->begin(); idx != stream->end(); ++idx)
        ss << "(" << *idx << ")";

        BOOST_CHECK(std::equal(stream->begin(), stream->end(), Grams<N>::grams[i].ids));
            //"Mismatched id in CheckOutput<" << (int)N << ">: " << ss.str();

        BOOST_CHECK_EQUAL(stream->Value().prob, Grams<N>::grams[i].prob);
/*                     "Mismatched probability in CheckOutput<"
                         << (int)N << ">, got " << stream->Value().prob
                         << ", expected " << Grams<N>::grams[i].prob;*/

        BOOST_CHECK_EQUAL(stream->Value().backoff, Grams<N>::grams[i].boff);
/*                     "Mismatched backoff in CheckOutput<"
                         << (int)N << ">, got " << stream->Value().backoff
                         << ", expected " << Grams<N>::grams[i].boff);*/
    }
    BOOST_CHECK_EQUAL(i , sizeof(Grams<N>::grams) / sizeof(Gram<N>));
/*                   "Did not get correct number of "
                       << (int)N << "-grams: expected "
                       << sizeof(Grams<N>::grams) / sizeof(Gram<N>)
                       << ", got " << i;*/
  }
};
}

BOOST_AUTO_TEST_CASE(BackoffReunificationTest) {
  util::stream::ChainConfig config;
  config.total_memory = 100;
  config.block_count = 1;

  util::stream::Chains prob_chains(3);
  config.entry_size = NGram<float>::TotalSize(1);
  prob_chains.push_back(config);
  prob_chains.back() >> WriteInput<1>();

  config.entry_size = NGram<float>::TotalSize(2);
  prob_chains.push_back(config);
  prob_chains.back() >> WriteInput<2>();

  config.entry_size = NGram<float>::TotalSize(3);
  prob_chains.push_back(config);
  prob_chains.back() >> WriteInput<3>();

  util::stream::Chains boff_chains(3);
  config.entry_size = sizeof(float);
  boff_chains.push_back(config);
  boff_chains.back() >> WriteBackoffs<1>();

  boff_chains.push_back(config);
  boff_chains.back() >> WriteBackoffs<2>();

  boff_chains.push_back(config);
  boff_chains.back() >> WriteBackoffs<3>();

  util::stream::ChainPositions prob_pos(prob_chains);
  util::stream::ChainPositions boff_pos(boff_chains);

  util::stream::Chains output_chains(3);
  for (std::size_t i = 0; i < 3; ++i) {
    config.entry_size = NGram<ProbBackoff>::TotalSize(i + 1);
    output_chains.push_back(config);
  }

  ReunifyBackoff(prob_pos, boff_pos, output_chains);

  output_chains[0] >> CheckOutput<1>();
  output_chains[1] >> CheckOutput<2>();
  output_chains[2] >> CheckOutput<3>();

  prob_chains >> util::stream::kRecycle;
  boff_chains >> util::stream::kRecycle;

  output_chains.Wait();
}
}
}
