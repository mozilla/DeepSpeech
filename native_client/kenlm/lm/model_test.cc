#include "lm/model.hh"

#include <cstdlib>
#include <cstring>

#define BOOST_TEST_MODULE ModelTest
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

// Apparently some Boost versions use templates and are pretty strict about types matching.
#define SLOPPY_CHECK_CLOSE(ref, value, tol) BOOST_CHECK_CLOSE(static_cast<double>(ref), static_cast<double>(value), static_cast<double>(tol));

namespace lm {
namespace ngram {

std::ostream &operator<<(std::ostream &o, const State &state) {
  o << "State length " << static_cast<unsigned int>(state.length) << ':';
  for (const WordIndex *i = state.words; i < state.words + state.length; ++i) {
    o << ' ' << *i;
  }
  return o;
}

namespace {

// Stupid bjam reverses the command line arguments randomly.
const char *TestLocation() {
  if (boost::unit_test::framework::master_test_suite().argc < 3) {
    return "test.arpa";
  }
  char **argv = boost::unit_test::framework::master_test_suite().argv;
  return argv[strstr(argv[1], "nounk") ? 2 : 1];
}
const char *TestNoUnkLocation() {
  if (boost::unit_test::framework::master_test_suite().argc < 3) {
    return "test_nounk.arpa";
  }
  char **argv = boost::unit_test::framework::master_test_suite().argv;
  return argv[strstr(argv[1], "nounk") ? 1 : 2];
}

template <class Model> State GetState(const Model &model, const char *word, const State &in) {
  WordIndex context[in.length + 1];
  context[0] = model.GetVocabulary().Index(word);
  std::copy(in.words, in.words + in.length, context + 1);
  State ret;
  model.GetState(context, context + in.length + 1, ret);
  return ret;
}

#define StartTest(word, ngram, score, indep_left) \
  ret = model.FullScore( \
      state, \
      model.GetVocabulary().Index(word), \
      out);\
  SLOPPY_CHECK_CLOSE(score, ret.prob, 0.001); \
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(ngram), ret.ngram_length); \
  BOOST_CHECK_GE(std::min<unsigned char>(ngram, 5 - 1), out.length); \
  BOOST_CHECK_EQUAL(indep_left, ret.independent_left); \
  BOOST_CHECK_EQUAL(out, GetState(model, word, state));

#define AppendTest(word, ngram, score, indep_left) \
  StartTest(word, ngram, score, indep_left) \
  state = out;

template <class M> void Starters(const M &model) {
  FullScoreReturn ret;
  Model::State state(model.BeginSentenceState());
  Model::State out;

  StartTest("looking", 2, -0.4846522, true);

  // , probability plus <s> backoff
  StartTest(",", 1, -1.383514 + -0.4149733, true);
  // <unk> probability plus <s> backoff
  StartTest("this_is_not_found", 1, -1.995635 + -0.4149733, true);
}

template <class M> void Continuation(const M &model) {
  FullScoreReturn ret;
  Model::State state(model.BeginSentenceState());
  Model::State out;

  AppendTest("looking", 2, -0.484652, true);
  AppendTest("on", 3, -0.348837, true);
  AppendTest("a", 4, -0.0155266, true);
  AppendTest("little", 5, -0.00306122, true);
  State preserve = state;
  AppendTest("the", 1, -4.04005, true);
  AppendTest("biarritz", 1, -1.9889, true);
  AppendTest("not_found", 1, -2.29666, true);
  AppendTest("more", 1, -1.20632 - 20.0, true);
  AppendTest(".", 2, -0.51363, true);
  AppendTest("</s>", 3, -0.0191651, true);
  BOOST_CHECK_EQUAL(0, state.length);

  state = preserve;
  AppendTest("more", 5, -0.00181395, true);
  BOOST_CHECK_EQUAL(4, state.length);
  AppendTest("loin", 5, -0.0432557, true);
  BOOST_CHECK_EQUAL(1, state.length);
}

template <class M> void Blanks(const M &model) {
  FullScoreReturn ret;
  State state(model.NullContextState());
  State out;
  AppendTest("also", 1, -1.687872, false);
  AppendTest("would", 2, -2, true);
  AppendTest("consider", 3, -3, true);
  State preserve = state;
  AppendTest("higher", 4, -4, true);
  AppendTest("looking", 5, -5, true);
  BOOST_CHECK_EQUAL(1, state.length);

  state = preserve;
  // also would consider not_found
  AppendTest("not_found", 1, -1.995635 - 7.0 - 0.30103, true);

  state = model.NullContextState();
  // higher looking is a blank.
  AppendTest("higher", 1, -1.509559, false);
  AppendTest("looking", 2, -1.285941 - 0.30103, false);

  State higher_looking = state;

  BOOST_CHECK_EQUAL(1, state.length);
  AppendTest("not_found", 1, -1.995635 - 0.4771212, true);

  state = higher_looking;
  // higher looking consider
  AppendTest("consider", 1, -1.687872 - 0.4771212, true);

  state = model.NullContextState();
  AppendTest("would", 1, -1.687872, false);
  BOOST_CHECK_EQUAL(1, state.length);
  AppendTest("consider", 2, -1.687872 -0.30103, false);
  BOOST_CHECK_EQUAL(2, state.length);
  AppendTest("higher", 3, -1.509559 - 0.30103, false);
  BOOST_CHECK_EQUAL(3, state.length);
  AppendTest("looking", 4, -1.285941 - 0.30103, false);
}

template <class M> void Unknowns(const M &model) {
  FullScoreReturn ret;
  State state(model.NullContextState());
  State out;

  AppendTest("not_found", 1, -1.995635, false);
  State preserve = state;
  AppendTest("not_found2", 2, -15.0, true);
  AppendTest("not_found3", 2, -15.0 - 2.0, true);

  state = preserve;
  AppendTest("however", 2, -4, true);
  AppendTest("not_found3", 3, -6, true);
}

template <class M> void MinimalState(const M &model) {
  FullScoreReturn ret;
  State state(model.NullContextState());
  State out;

  AppendTest("baz", 1, -6.535897, true);
  BOOST_CHECK_EQUAL(0, state.length);
  state = model.NullContextState();
  AppendTest("foo", 1, -3.141592, true);
  BOOST_CHECK_EQUAL(1, state.length);
  AppendTest("bar", 2, -6.0, true);
  // Has to include the backoff weight.
  BOOST_CHECK_EQUAL(1, state.length);
  AppendTest("bar", 1, -2.718281 + 3.0, true);
  BOOST_CHECK_EQUAL(1, state.length);

  state = model.NullContextState();
  AppendTest("to", 1, -1.687872, false);
  AppendTest("look", 2, -0.2922095, true);
  BOOST_CHECK_EQUAL(2, state.length);
  AppendTest("a", 3, -7, true);
}

template <class M> void ExtendLeftTest(const M &model) {
  State right;
  FullScoreReturn little(model.FullScore(model.NullContextState(), model.GetVocabulary().Index("little"), right));
  const float kLittleProb = -1.285941;
  SLOPPY_CHECK_CLOSE(kLittleProb, little.prob, 0.001);
  unsigned char next_use;
  float backoff_out[4];

  FullScoreReturn extend_none(model.ExtendLeft(NULL, NULL, NULL, little.extend_left, 1, NULL, next_use));
  BOOST_CHECK_EQUAL(0, next_use);
  BOOST_CHECK_EQUAL(little.extend_left, extend_none.extend_left);
  SLOPPY_CHECK_CLOSE(little.prob - little.rest, extend_none.prob, 0.001);
  BOOST_CHECK_EQUAL(1, extend_none.ngram_length);

  const WordIndex a = model.GetVocabulary().Index("a");
  float backoff_in = 3.14;
  // a little
  FullScoreReturn extend_a(model.ExtendLeft(&a, &a + 1, &backoff_in, little.extend_left, 1, backoff_out, next_use));
  BOOST_CHECK_EQUAL(1, next_use);
  SLOPPY_CHECK_CLOSE(-0.69897, backoff_out[0], 0.001);
  SLOPPY_CHECK_CLOSE(-0.09132547 - little.rest, extend_a.prob, 0.001);
  BOOST_CHECK_EQUAL(2, extend_a.ngram_length);
  BOOST_CHECK(!extend_a.independent_left);

  const WordIndex on = model.GetVocabulary().Index("on");
  FullScoreReturn extend_on(model.ExtendLeft(&on, &on + 1, &backoff_in, extend_a.extend_left, 2, backoff_out, next_use));
  BOOST_CHECK_EQUAL(1, next_use);
  SLOPPY_CHECK_CLOSE(-0.4771212, backoff_out[0], 0.001);
  SLOPPY_CHECK_CLOSE(-0.0283603 - (extend_a.rest + little.rest), extend_on.prob, 0.001);
  BOOST_CHECK_EQUAL(3, extend_on.ngram_length);
  BOOST_CHECK(!extend_on.independent_left);

  const WordIndex both[2] = {a, on};
  float backoff_in_arr[4];
  FullScoreReturn extend_both(model.ExtendLeft(both, both + 2, backoff_in_arr, little.extend_left, 1, backoff_out, next_use));
  BOOST_CHECK_EQUAL(2, next_use);
  SLOPPY_CHECK_CLOSE(-0.69897, backoff_out[0], 0.001);
  SLOPPY_CHECK_CLOSE(-0.4771212, backoff_out[1], 0.001);
  SLOPPY_CHECK_CLOSE(-0.0283603 - little.rest, extend_both.prob, 0.001);
  BOOST_CHECK_EQUAL(3, extend_both.ngram_length);
  BOOST_CHECK(!extend_both.independent_left);
  BOOST_CHECK_EQUAL(extend_on.extend_left, extend_both.extend_left);
}

#define StatelessTest(word, provide, ngram, score) \
  ret = model.FullScoreForgotState(indices + num_words - word, indices + num_words - word + provide, indices[num_words - word - 1], state); \
  SLOPPY_CHECK_CLOSE(score, ret.prob, 0.001); \
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(ngram), ret.ngram_length); \
  model.GetState(indices + num_words - word, indices + num_words - word + provide, before); \
  ret = model.FullScore(before, indices[num_words - word - 1], out); \
  BOOST_CHECK(state == out); \
  SLOPPY_CHECK_CLOSE(score, ret.prob, 0.001); \
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(ngram), ret.ngram_length);

template <class M> void Stateless(const M &model) {
  const char *words[] = {"<s>", "looking", "on", "a", "little", "the", "biarritz", "not_found", "more", ".", "</s>"};
  const size_t num_words = sizeof(words) / sizeof(const char*);
  // Silience "array subscript is above array bounds" when extracting end pointer.
  WordIndex indices[num_words + 1];
  for (unsigned int i = 0; i < num_words; ++i) {
    indices[num_words - 1 - i] = model.GetVocabulary().Index(words[i]);
  }
  FullScoreReturn ret;
  State state, out, before;

  ret = model.FullScoreForgotState(indices + num_words - 1, indices + num_words, indices[num_words - 2], state);
  SLOPPY_CHECK_CLOSE(-0.484652, ret.prob, 0.001);
  StatelessTest(1, 1, 2, -0.484652);

  // looking
  StatelessTest(1, 2, 2, -0.484652);
  // on
  AppendTest("on", 3, -0.348837, true);
  StatelessTest(2, 3, 3, -0.348837);
  StatelessTest(2, 2, 3, -0.348837);
  StatelessTest(2, 1, 2, -0.4638903);
  // a
  StatelessTest(3, 4, 4, -0.0155266);
  // little
  AppendTest("little", 5, -0.00306122, true);
  StatelessTest(4, 5, 5, -0.00306122);
  // the
  AppendTest("the", 1, -4.04005, true);
  StatelessTest(5, 5, 1, -4.04005);
  // No context of the.
  StatelessTest(5, 0, 1, -1.687872);
  // biarritz
  StatelessTest(6, 1, 1, -1.9889);
  // not found
  StatelessTest(7, 1, 1, -2.29666);
  StatelessTest(7, 0, 1, -1.995635);

  WordIndex unk[1];
  unk[0] = 0;
  model.GetState(unk, unk + 1, state);
  BOOST_CHECK_EQUAL(1, state.length);
  BOOST_CHECK_EQUAL(static_cast<WordIndex>(0), state.words[0]);
}

template <class M> void NoUnkCheck(const M &model) {
  WordIndex unk_index = 0;
  State state;

  FullScoreReturn ret = model.FullScoreForgotState(&unk_index, &unk_index + 1, unk_index, state);
  SLOPPY_CHECK_CLOSE(-100.0, ret.prob, 0.001);
}

template <class M> void Everything(const M &m) {
  Starters(m);
  Continuation(m);
  Blanks(m);
  Unknowns(m);
  MinimalState(m);
  ExtendLeftTest(m);
  Stateless(m);
}

class ExpectEnumerateVocab : public EnumerateVocab {
  public:
    ExpectEnumerateVocab() {}

    void Add(WordIndex index, const StringPiece &str) {
      BOOST_CHECK_EQUAL(seen.size(), index);
      seen.push_back(std::string(str.data(), str.length()));
    }

    void Check(const base::Vocabulary &vocab) {
      BOOST_CHECK_EQUAL(37ULL, seen.size());
      BOOST_REQUIRE(!seen.empty());
      BOOST_CHECK_EQUAL("<unk>", seen[0]);
      for (WordIndex i = 0; i < seen.size(); ++i) {
        BOOST_CHECK_EQUAL(i, vocab.Index(seen[i]));
      }
    }

    void Clear() {
      seen.clear();
    }

    std::vector<std::string> seen;
};

template <class ModelT> void LoadingTest() {
  Config config;
  config.arpa_complain = Config::NONE;
  config.messages = NULL;
  config.probing_multiplier = 2.0;
  {
    ExpectEnumerateVocab enumerate;
    config.enumerate_vocab = &enumerate;
    ModelT m(TestLocation(), config);
    enumerate.Check(m.GetVocabulary());
    BOOST_CHECK_EQUAL((WordIndex)37, m.GetVocabulary().Bound());
    Everything(m);
  }
  {
    ExpectEnumerateVocab enumerate;
    config.enumerate_vocab = &enumerate;
    ModelT m(TestNoUnkLocation(), config);
    enumerate.Check(m.GetVocabulary());
    BOOST_CHECK_EQUAL((WordIndex)37, m.GetVocabulary().Bound());
    NoUnkCheck(m);
  }
}

BOOST_AUTO_TEST_CASE(probing) {
  LoadingTest<Model>();
}
BOOST_AUTO_TEST_CASE(trie) {
  LoadingTest<TrieModel>();
}
BOOST_AUTO_TEST_CASE(quant_trie) {
  LoadingTest<QuantTrieModel>();
}
BOOST_AUTO_TEST_CASE(bhiksha_trie) {
  LoadingTest<ArrayTrieModel>();
}
BOOST_AUTO_TEST_CASE(quant_bhiksha_trie) {
  LoadingTest<QuantArrayTrieModel>();
}

template <class ModelT> void BinaryTest(Config::WriteMethod write_method) {
  Config config;
  config.write_mmap = "test.binary";
  config.messages = NULL;
  config.write_method = write_method;
  ExpectEnumerateVocab enumerate;
  config.enumerate_vocab = &enumerate;

  {
    ModelT copy_model(TestLocation(), config);
    enumerate.Check(copy_model.GetVocabulary());
    enumerate.Clear();
    Everything(copy_model);
  }

  config.write_mmap = NULL;

  ModelType type;
  BOOST_REQUIRE(RecognizeBinary("test.binary", type));
  BOOST_CHECK_EQUAL(ModelT::kModelType, type);

  {
    ModelT binary("test.binary", config);
    enumerate.Check(binary.GetVocabulary());
    Everything(binary);
  }
  unlink("test.binary");

  // Now test without <unk>.
  config.write_mmap = "test_nounk.binary";
  config.messages = NULL;
  enumerate.Clear();
  {
    ModelT copy_model(TestNoUnkLocation(), config);
    enumerate.Check(copy_model.GetVocabulary());
    enumerate.Clear();
    NoUnkCheck(copy_model);
  }
  config.write_mmap = NULL;
  {
    ModelT binary(TestNoUnkLocation(), config);
    enumerate.Check(binary.GetVocabulary());
    NoUnkCheck(binary);
  }
  unlink("test_nounk.binary");
}

template <class ModelT> void BinaryTest() {
  BinaryTest<ModelT>(Config::WRITE_MMAP);
  BinaryTest<ModelT>(Config::WRITE_AFTER);
}

BOOST_AUTO_TEST_CASE(write_and_read_probing) {
  BinaryTest<ProbingModel>();
}
BOOST_AUTO_TEST_CASE(write_and_read_rest_probing) {
  BinaryTest<RestProbingModel>();
}
BOOST_AUTO_TEST_CASE(write_and_read_trie) {
  BinaryTest<TrieModel>();
}
BOOST_AUTO_TEST_CASE(write_and_read_quant_trie) {
  BinaryTest<QuantTrieModel>();
}
BOOST_AUTO_TEST_CASE(write_and_read_array_trie) {
  BinaryTest<ArrayTrieModel>();
}
BOOST_AUTO_TEST_CASE(write_and_read_quant_array_trie) {
  BinaryTest<QuantArrayTrieModel>();
}

BOOST_AUTO_TEST_CASE(rest_max) {
  Config config;
  config.arpa_complain = Config::NONE;
  config.messages = NULL;

  RestProbingModel model(TestLocation(), config);
  State state, out;
  FullScoreReturn ret(model.FullScore(model.NullContextState(), model.GetVocabulary().Index("."), state));
  SLOPPY_CHECK_CLOSE(-0.2705918, ret.rest, 0.001);
  SLOPPY_CHECK_CLOSE(-0.01916512, model.FullScore(state, model.GetVocabulary().EndSentence(), out).rest, 0.001);
}

} // namespace
} // namespace ngram
} // namespace lm
