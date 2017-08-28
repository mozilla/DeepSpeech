#include "lm/interpolate/tune_instances.hh"

#include "util/file.hh"
#include "util/file_stream.hh"
#include "util/stream/chain.hh"
#include "util/stream/config.hh"
#include "util/stream/typed_stream.hh"
#include "util/string_piece.hh"

#define BOOST_TEST_MODULE InstanceTest
#include <boost/test/unit_test.hpp>

#include <vector>

#include <math.h>

namespace lm { namespace interpolate { namespace {

BOOST_AUTO_TEST_CASE(Toy) {
  util::scoped_fd test_input(util::MakeTemp("temporary"));
  util::FileStream(test_input.get()) << "c\n";

  StringPiece dir("../common/test_data/");
  if (boost::unit_test::framework::master_test_suite().argc == 2) {
    StringPiece zero_file(boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_REQUIRE(zero_file.size() > strlen("toy0.1"));
    BOOST_REQUIRE_EQUAL("toy0.1", StringPiece(zero_file.data() + zero_file.size() - 6, 6));
    dir = StringPiece(zero_file.data(), zero_file.size() - 6);
  }

  std::vector<StringPiece> model_names;
  std::string full0 = std::string(dir.data(), dir.size()) + "toy0";
  std::string full1 = std::string(dir.data(), dir.size()) + "toy1";
  model_names.push_back(full0);
  model_names.push_back(full1);

  // Tiny buffer sizes.
  InstancesConfig config;
  config.model_read_chain_mem = 100;
  config.extension_write_chain_mem = 100;
  config.lazy_memory = 100;
  config.sort.temp_prefix = "temporary";
  config.sort.buffer_size = 100;
  config.sort.total_memory = 1024;

  util::SeekOrThrow(test_input.get(), 0);

  Instances inst(test_input.release(), model_names, config);

  BOOST_CHECK_EQUAL(1, inst.BOS());
  const Matrix &ln_unigrams = inst.LNUnigrams();

  // <unk>=0
  BOOST_CHECK_CLOSE(-0.90309 * M_LN10, ln_unigrams(0, 0), 0.001);
  BOOST_CHECK_CLOSE(-1 * M_LN10, ln_unigrams(0, 1), 0.001);
  // <s>=1 doesn't matter as long as it doesn't cause NaNs.
  BOOST_CHECK(!isnan(ln_unigrams(1, 0)));
  BOOST_CHECK(!isnan(ln_unigrams(1, 1)));
  // a = 2
  BOOST_CHECK_CLOSE(-0.46943438 * M_LN10, ln_unigrams(2, 0), 0.001);
  BOOST_CHECK_CLOSE(-0.6146491 * M_LN10, ln_unigrams(2, 1), 0.001);
  // </s> = 3
  BOOST_CHECK_CLOSE(-0.5720968 * M_LN10, ln_unigrams(3, 0), 0.001);
  BOOST_CHECK_CLOSE(-0.6146491 * M_LN10, ln_unigrams(3, 1), 0.001);
  // c = 4
  BOOST_CHECK_CLOSE(-0.90309 * M_LN10, ln_unigrams(4, 0), 0.001); // <unk>
  BOOST_CHECK_CLOSE(-0.7659168 * M_LN10, ln_unigrams(4, 1), 0.001);
  // too lazy to do b = 5.

  // Two instances:
  // <s> predicts c
  // <s> c predicts </s>
  BOOST_REQUIRE_EQUAL(2, inst.NumInstances());
  BOOST_CHECK_CLOSE(-0.30103 * M_LN10, inst.LNBackoffs(0)(0), 0.001);
  BOOST_CHECK_CLOSE(-0.30103 * M_LN10, inst.LNBackoffs(0)(1), 0.001);


  // Backoffs of <s> c
  BOOST_CHECK_CLOSE(0.0, inst.LNBackoffs(1)(0), 0.001);
  BOOST_CHECK_CLOSE((-0.30103 - 0.30103) * M_LN10, inst.LNBackoffs(1)(1), 0.001);

  util::stream::Chain extensions(util::stream::ChainConfig(inst.ReadExtensionsEntrySize(), 2, 300));
  inst.ReadExtensions(extensions);
  util::stream::TypedStream<Extension> stream(extensions.Add());
  extensions >> util::stream::kRecycle;

  // The extensions are (in order of instance, vocab id, and model as they should be sorted):
  // <s> a from both models 0 and 1 (so two instances)
  // <s> c from model 1
  // <s> b from model 0
  // c </s> from model 1
  // Magic probabilities come from querying the models directly.

  // <s> a from model 0
  BOOST_REQUIRE(stream);
  BOOST_CHECK_EQUAL(0, stream->instance);
  BOOST_CHECK_EQUAL(2 /* a */, stream->word);
  BOOST_CHECK_EQUAL(0, stream->model);
  BOOST_CHECK_CLOSE(-0.37712017 * M_LN10, stream->ln_prob, 0.001);

  // <s> a from model 1
  BOOST_REQUIRE(++stream);
  BOOST_CHECK_EQUAL(0, stream->instance);
  BOOST_CHECK_EQUAL(2 /* a */, stream->word);
  BOOST_CHECK_EQUAL(1, stream->model);
  BOOST_CHECK_CLOSE(-0.4301247 * M_LN10, stream->ln_prob, 0.001);

  // <s> c from model 1
  BOOST_REQUIRE(++stream);
  BOOST_CHECK_EQUAL(0, stream->instance);
  BOOST_CHECK_EQUAL(4 /* c */, stream->word);
  BOOST_CHECK_EQUAL(1, stream->model);
  BOOST_CHECK_CLOSE(-0.4740302 * M_LN10, stream->ln_prob, 0.001);

  // <s> b from model 0
  BOOST_REQUIRE(++stream);
  BOOST_CHECK_EQUAL(0, stream->instance);
  BOOST_CHECK_EQUAL(5 /* b */, stream->word);
  BOOST_CHECK_EQUAL(0, stream->model);
  BOOST_CHECK_CLOSE(-0.41574955 * M_LN10, stream->ln_prob, 0.001);

  // c </s> from model 1
  BOOST_REQUIRE(++stream);
  BOOST_CHECK_EQUAL(1, stream->instance);
  BOOST_CHECK_EQUAL(3 /* </s> */, stream->word);
  BOOST_CHECK_EQUAL(1, stream->model);
  BOOST_CHECK_CLOSE(-0.09113217 * M_LN10, stream->ln_prob, 0.001);

  BOOST_CHECK(!++stream);
}

}}} // namespaces
