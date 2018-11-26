#include "lm/left.hh"
#include "lm/model.hh"

#include "util/tokenize_piece.hh"

#include <vector>

#define BOOST_TEST_MODULE LeftTest
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

namespace lm {
namespace ngram {
namespace {

#define Term(word) score.Terminal(m.GetVocabulary().Index(word));
#define VCheck(word, value) BOOST_CHECK_EQUAL(m.GetVocabulary().Index(word), value);

// Apparently some Boost versions use templates and are pretty strict about types matching.
#define SLOPPY_CHECK_CLOSE(ref, value, tol) BOOST_CHECK_CLOSE(static_cast<double>(ref), static_cast<double>(value), static_cast<double>(tol));

template <class M> void Short(const M &m) {
  ChartState base;
  {
    RuleScore<M> score(m, base);
    Term("more");
    Term("loin");
    SLOPPY_CHECK_CLOSE(-1.206319 - 0.3561665, score.Finish(), 0.001);
  }
  BOOST_CHECK(base.left.full);
  BOOST_CHECK_EQUAL(2, base.left.length);
  BOOST_CHECK_EQUAL(1, base.right.length);
  VCheck("loin", base.right.words[0]);

  ChartState more_left;
  {
    RuleScore<M> score(m, more_left);
    Term("little");
    score.NonTerminal(base, -1.206319 - 0.3561665);
    // p(little more loin | null context)
    SLOPPY_CHECK_CLOSE(-1.56538, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(3, more_left.left.length);
  BOOST_CHECK_EQUAL(1, more_left.right.length);
  VCheck("loin", more_left.right.words[0]);
  BOOST_CHECK(more_left.left.full);

  ChartState shorter;
  {
    RuleScore<M> score(m, shorter);
    Term("to");
    score.NonTerminal(base, -1.206319 - 0.3561665);
    SLOPPY_CHECK_CLOSE(-0.30103 - 1.687872 - 1.206319 - 0.3561665, score.Finish(), 0.01);
  }
  BOOST_CHECK_EQUAL(1, shorter.left.length);
  BOOST_CHECK_EQUAL(1, shorter.right.length);
  VCheck("loin", shorter.right.words[0]);
  BOOST_CHECK(shorter.left.full);
}

template <class M> void Charge(const M &m) {
  ChartState base;
  {
    RuleScore<M> score(m, base);
    Term("on");
    Term("more");
    SLOPPY_CHECK_CLOSE(-1.509559 -0.4771212 -1.206319, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(1, base.left.length);
  BOOST_CHECK_EQUAL(1, base.right.length);
  VCheck("more", base.right.words[0]);
  BOOST_CHECK(base.left.full);

  ChartState extend;
  {
    RuleScore<M> score(m, extend);
    Term("looking");
    score.NonTerminal(base, -1.509559 -0.4771212 -1.206319);
    SLOPPY_CHECK_CLOSE(-3.91039, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(2, extend.left.length);
  BOOST_CHECK_EQUAL(1, extend.right.length);
  VCheck("more", extend.right.words[0]);
  BOOST_CHECK(extend.left.full);

  ChartState tobos;
  {
    RuleScore<M> score(m, tobos);
    score.BeginSentence();
    score.NonTerminal(extend, -3.91039);
    SLOPPY_CHECK_CLOSE(-3.471169, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(0, tobos.left.length);
  BOOST_CHECK_EQUAL(1, tobos.right.length);
}

template <class M> float LeftToRight(const M &m, const std::vector<WordIndex> &words, bool begin_sentence = false) {
  float ret = 0.0;
  State right = begin_sentence ? m.BeginSentenceState() : m.NullContextState();
  for (std::vector<WordIndex>::const_iterator i = words.begin(); i != words.end(); ++i) {
    State copy(right);
    ret += m.Score(copy, *i, right);
  }
  return ret;
}

template <class M> float RightToLeft(const M &m, const std::vector<WordIndex> &words, bool begin_sentence = false) {
  float ret = 0.0;
  ChartState state;
  state.left.length = 0;
  state.right.length = 0;
  state.left.full = false;
  for (std::vector<WordIndex>::const_reverse_iterator i = words.rbegin(); i != words.rend(); ++i) {
    ChartState copy(state);
    RuleScore<M> score(m, state);
    score.Terminal(*i);
    score.NonTerminal(copy, ret);
    ret = score.Finish();
  }
  if (begin_sentence) {
    ChartState copy(state);
    RuleScore<M> score(m, state);
    score.BeginSentence();
    score.NonTerminal(copy, ret);
    ret = score.Finish();
  }
  return ret;
}

template <class M> float TreeMiddle(const M &m, const std::vector<WordIndex> &words, bool begin_sentence = false) {
  std::vector<std::pair<ChartState, float> > states(words.size());
  for (unsigned int i = 0; i < words.size(); ++i) {
    RuleScore<M> score(m, states[i].first);
    score.Terminal(words[i]);
    states[i].second = score.Finish();
  }
  while (states.size() > 1) {
    std::vector<std::pair<ChartState, float> > upper((states.size() + 1) / 2);
    for (unsigned int i = 0; i < states.size() / 2; ++i) {
      RuleScore<M> score(m, upper[i].first);
      score.NonTerminal(states[i*2].first, states[i*2].second);
      score.NonTerminal(states[i*2+1].first, states[i*2+1].second);
      upper[i].second = score.Finish();
    }
    if (states.size() % 2) {
      upper.back() = states.back();
    }
    std::swap(states, upper);
  }

  if (states.empty()) return 0.0;

  if (begin_sentence) {
    ChartState ignored;
    RuleScore<M> score(m, ignored);
    score.BeginSentence();
    score.NonTerminal(states.front().first, states.front().second);
    return score.Finish();
  } else {
    return states.front().second;
  }

}

template <class M> void LookupVocab(const M &m, const StringPiece &str, std::vector<WordIndex> &out) {
  out.clear();
  for (util::TokenIter<util::SingleCharacter, true> i(str, ' '); i; ++i) {
    out.push_back(m.GetVocabulary().Index(*i));
  }
}

#define TEXT_TEST(str) \
  LookupVocab(m, str, words); \
  expect = LeftToRight(m, words, rest); \
  SLOPPY_CHECK_CLOSE(expect, RightToLeft(m, words, rest), 0.001); \
  SLOPPY_CHECK_CLOSE(expect, TreeMiddle(m, words, rest), 0.001); \

// Build sentences, or parts thereof, from right to left.
template <class M> void GrowBig(const M &m, bool rest = false) {
  std::vector<WordIndex> words;
  float expect;
  TEXT_TEST("in biarritz watching considering looking . on a little more loin also would consider higher to look good unknown the screening foo bar , unknown however unknown </s>");
  TEXT_TEST("on a little more loin also would consider higher to look good unknown the screening foo bar , unknown however unknown </s>");
  TEXT_TEST("on a little more loin also would consider higher to look good");
  TEXT_TEST("more loin also would consider higher to look good");
  TEXT_TEST("more loin also would consider higher to look");
  TEXT_TEST("also would consider higher to look");
  TEXT_TEST("also would consider higher");
  TEXT_TEST("would consider higher to look");
  TEXT_TEST("consider higher to look");
  TEXT_TEST("consider higher to");
  TEXT_TEST("consider higher");
}

template <class M> void GrowSmall(const M &m, bool rest = false) {
  std::vector<WordIndex> words;
  float expect;
  TEXT_TEST("in biarritz watching considering looking . </s>");
  TEXT_TEST("in biarritz watching considering looking .");
  TEXT_TEST("in biarritz");
}

template <class M> void AlsoWouldConsiderHigher(const M &m) {
  ChartState also;
  {
    RuleScore<M> score(m, also);
    score.Terminal(m.GetVocabulary().Index("also"));
    SLOPPY_CHECK_CLOSE(-1.687872, score.Finish(), 0.001);
  }
  ChartState would;
  {
    RuleScore<M> score(m, would);
    score.Terminal(m.GetVocabulary().Index("would"));
    SLOPPY_CHECK_CLOSE(-1.687872, score.Finish(), 0.001);
  }
  ChartState combine_also_would;
  {
    RuleScore<M> score(m, combine_also_would);
    score.NonTerminal(also, -1.687872);
    score.NonTerminal(would, -1.687872);
    SLOPPY_CHECK_CLOSE(-1.687872 - 2.0, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(2, combine_also_would.right.length);

  ChartState also_would;
  {
    RuleScore<M> score(m, also_would);
    score.Terminal(m.GetVocabulary().Index("also"));
    score.Terminal(m.GetVocabulary().Index("would"));
    SLOPPY_CHECK_CLOSE(-1.687872 - 2.0, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(2, also_would.right.length);

  ChartState consider;
  {
    RuleScore<M> score(m, consider);
    score.Terminal(m.GetVocabulary().Index("consider"));
    SLOPPY_CHECK_CLOSE(-1.687872, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(1, consider.left.length);
  BOOST_CHECK_EQUAL(1, consider.right.length);
  BOOST_CHECK(!consider.left.full);

  ChartState higher;
  float higher_score;
  {
    RuleScore<M> score(m, higher);
    score.Terminal(m.GetVocabulary().Index("higher"));
    higher_score = score.Finish();
  }
  SLOPPY_CHECK_CLOSE(-1.509559, higher_score, 0.001);
  BOOST_CHECK_EQUAL(1, higher.left.length);
  BOOST_CHECK_EQUAL(1, higher.right.length);
  BOOST_CHECK(!higher.left.full);
  VCheck("higher", higher.right.words[0]);
  SLOPPY_CHECK_CLOSE(-0.30103, higher.right.backoff[0], 0.001);

  ChartState consider_higher;
  {
    RuleScore<M> score(m, consider_higher);
    score.NonTerminal(consider, -1.687872);
    score.NonTerminal(higher, higher_score);
    SLOPPY_CHECK_CLOSE(-1.509559 - 1.687872 - 0.30103, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(2, consider_higher.left.length);
  BOOST_CHECK(!consider_higher.left.full);

  ChartState full;
  {
    RuleScore<M> score(m, full);
    score.NonTerminal(combine_also_would, -1.687872 - 2.0);
    score.NonTerminal(consider_higher, -1.509559 - 1.687872 - 0.30103);
    SLOPPY_CHECK_CLOSE(-10.6879, score.Finish(), 0.001);
  }
  BOOST_CHECK_EQUAL(4, full.right.length);
}

#define CHECK_SCORE(str, val) \
{ \
  float got = val; \
  std::vector<WordIndex> indices; \
  LookupVocab(m, str, indices); \
  SLOPPY_CHECK_CLOSE(LeftToRight(m, indices), got, 0.001); \
}

template <class M> void FullGrow(const M &m) {
  std::vector<WordIndex> words;
  LookupVocab(m, "in biarritz watching considering looking . </s>", words);

  ChartState lexical[7];
  float lexical_scores[7];
  for (unsigned int i = 0; i < 7; ++i) {
    RuleScore<M> score(m, lexical[i]);
    score.Terminal(words[i]);
    lexical_scores[i] = score.Finish();
  }
  CHECK_SCORE("in", lexical_scores[0]);
  CHECK_SCORE("biarritz", lexical_scores[1]);
  CHECK_SCORE("watching", lexical_scores[2]);
  CHECK_SCORE("</s>", lexical_scores[6]);

  ChartState l1[4];
  float l1_scores[4];
  {
    RuleScore<M> score(m, l1[0]);
    score.NonTerminal(lexical[0], lexical_scores[0]);
    score.NonTerminal(lexical[1], lexical_scores[1]);
    CHECK_SCORE("in biarritz", l1_scores[0] = score.Finish());
  }
  {
    RuleScore<M> score(m, l1[1]);
    score.NonTerminal(lexical[2], lexical_scores[2]);
    score.NonTerminal(lexical[3], lexical_scores[3]);
    CHECK_SCORE("watching considering", l1_scores[1] = score.Finish());
  }
  {
    RuleScore<M> score(m, l1[2]);
    score.NonTerminal(lexical[4], lexical_scores[4]);
    score.NonTerminal(lexical[5], lexical_scores[5]);
    CHECK_SCORE("looking .", l1_scores[2] = score.Finish());
  }
  BOOST_CHECK_EQUAL(l1[2].left.length, 1);
  l1[3] = lexical[6];
  l1_scores[3] = lexical_scores[6];

  ChartState l2[2];
  float l2_scores[2];
  {
    RuleScore<M> score(m, l2[0]);
    score.NonTerminal(l1[0], l1_scores[0]);
    score.NonTerminal(l1[1], l1_scores[1]);
    CHECK_SCORE("in biarritz watching considering", l2_scores[0] = score.Finish());
  }
  {
    RuleScore<M> score(m, l2[1]);
    score.NonTerminal(l1[2], l1_scores[2]);
    score.NonTerminal(l1[3], l1_scores[3]);
    CHECK_SCORE("looking . </s>", l2_scores[1] = score.Finish());
  }
  BOOST_CHECK_EQUAL(l2[1].left.length, 1);
  BOOST_CHECK(l2[1].left.full);

  ChartState top;
  {
    RuleScore<M> score(m, top);
    score.NonTerminal(l2[0], l2_scores[0]);
    score.NonTerminal(l2[1], l2_scores[1]);
    CHECK_SCORE("in biarritz watching considering looking . </s>", score.Finish());
  }
}

const char *FileLocation() {
  if (boost::unit_test::framework::master_test_suite().argc < 2) {
    return "test.arpa";
  }
  return boost::unit_test::framework::master_test_suite().argv[1];
}

template <class M> void Everything() {
  Config config;
  config.messages = NULL;
  M m(FileLocation(), config);

  Short(m);
  Charge(m);
  GrowBig(m);
  AlsoWouldConsiderHigher(m);
  GrowSmall(m);
  FullGrow(m);
}

BOOST_AUTO_TEST_CASE(ProbingAll) {
  Everything<Model>();
}
BOOST_AUTO_TEST_CASE(TrieAll) {
  Everything<TrieModel>();
}
BOOST_AUTO_TEST_CASE(QuantTrieAll) {
  Everything<QuantTrieModel>();
}
BOOST_AUTO_TEST_CASE(ArrayQuantTrieAll) {
  Everything<QuantArrayTrieModel>();
}
BOOST_AUTO_TEST_CASE(ArrayTrieAll) {
  Everything<ArrayTrieModel>();
}

BOOST_AUTO_TEST_CASE(RestProbing) {
  Config config;
  config.messages = NULL;
  RestProbingModel m(FileLocation(), config);
  GrowBig(m, true);
}

} // namespace
} // namespace ngram
} // namespace lm
