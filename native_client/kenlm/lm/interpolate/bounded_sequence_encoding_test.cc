#include "lm/interpolate/bounded_sequence_encoding.hh"

#include "util/scoped.hh"

#define BOOST_TEST_MODULE BoundedSequenceEncodingTest
#include <boost/test/unit_test.hpp>

namespace lm {
namespace interpolate {
namespace {

void ExhaustiveTest(unsigned char *bound_begin, unsigned char *bound_end) {
  BoundedSequenceEncoding enc(bound_begin, bound_end);
  util::scoped_malloc backing(util::MallocOrThrow(enc.EncodedLength()));
  std::vector<unsigned char> values(bound_end - bound_begin),
      out(bound_end - bound_begin);
  while (true) {
    enc.Encode(&values[0], backing.get());
    enc.Decode(backing.get(), &out[0]);
    for (std::size_t i = 0; i != values.size(); ++i) {
      BOOST_CHECK_EQUAL(values[i], out[i]);
    }
    for (std::size_t i = 0;; ++i) {
      if (i == values.size()) return;
      ++values[i];
      if (values[i] < bound_begin[i]) break;
      values[i] = 0;
    }
  }
}

void CheckEncodeDecode(unsigned char *bounds, unsigned char *input,
                       unsigned char *output, std::size_t len) {
  BoundedSequenceEncoding encoder(bounds, bounds + len);
  util::scoped_malloc backing(util::MallocOrThrow(encoder.EncodedLength()));

  encoder.Encode(input, backing.get());
  encoder.Decode(backing.get(), output);

  for (std::size_t i = 0; i < len; ++i) {
    BOOST_CHECK_EQUAL(input[i], output[i]);
  }
}

BOOST_AUTO_TEST_CASE(Exhaustive) {
  unsigned char bounds[] = {5, 2, 3, 9, 7, 20, 8};
  ExhaustiveTest(bounds, bounds + sizeof(bounds) / sizeof(unsigned char));
}

BOOST_AUTO_TEST_CASE(LessThan64) {
  unsigned char bounds[] = {255, 255, 255, 255, 255, 255, 255, 3};
  unsigned char input[] = {172, 183, 254, 187, 96, 87, 65, 2};
  unsigned char output[] = {0, 0, 0, 0, 0, 0, 0, 0};

  std::size_t len = sizeof(bounds) / sizeof(unsigned char);
  assert(sizeof(input) / sizeof(unsigned char) == len);
  assert(sizeof(output) / sizeof(unsigned char) == len);

  CheckEncodeDecode(bounds, input, output, len);
}

BOOST_AUTO_TEST_CASE(Exactly64) {
  unsigned char bounds[] = {255, 255, 255, 255, 255, 255, 255, 255};
  unsigned char input[] = {172, 183, 254, 187, 96, 87, 65, 16};
  unsigned char output[] = {0, 0, 0, 0, 0, 0, 0, 0};

  std::size_t len = sizeof(bounds) / sizeof(unsigned char);
  assert(sizeof(input) / sizeof(unsigned char) == len);
  assert(sizeof(output) / sizeof(unsigned char) == len);

  CheckEncodeDecode(bounds, input, output, len);
}

BOOST_AUTO_TEST_CASE(MoreThan64) {
  unsigned char bounds[] = {255, 255, 255, 255, 255, 255, 255, 255, 255};
  unsigned char input[] = {172, 183, 254, 187, 96, 87, 65, 16, 137};
  unsigned char output[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::size_t len = sizeof(bounds) / sizeof(unsigned char);
  assert(sizeof(input) / sizeof(unsigned char) == len);
  assert(sizeof(output) / sizeof(unsigned char) == len);

  CheckEncodeDecode(bounds, input, output, len);
}

}}} // namespaces
