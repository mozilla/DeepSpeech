#include "lm/model.hh"

#include "lm/blank.hh"
#include "lm/lm_exception.hh"
#include "lm/search_hashed.hh"
#include "lm/search_trie.hh"
#include "lm/read_arpa.hh"
#include "util/have.hh"
#include "util/murmur_hash.hh"

#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>
#include <limits>

namespace lm {
namespace ngram {
namespace detail {

template <class Search, class VocabularyT> const ModelType GenericModel<Search, VocabularyT>::kModelType = Search::kModelType;

template <class Search, class VocabularyT> uint64_t GenericModel<Search, VocabularyT>::Size(const std::vector<uint64_t> &counts, const Config &config) {
  return VocabularyT::Size(counts[0], config) + Search::Size(counts, config);
}

template <class Search, class VocabularyT> void GenericModel<Search, VocabularyT>::SetupMemory(void *base, const std::vector<uint64_t> &counts, const Config &config) {
  size_t goal_size = util::CheckOverflow(Size(counts, config));
  uint8_t *start = static_cast<uint8_t*>(base);
  size_t allocated = VocabularyT::Size(counts[0], config);
  vocab_.SetupMemory(start, allocated, counts[0], config);
  start += allocated;
  start = search_.SetupMemory(start, counts, config);
  if (static_cast<std::size_t>(start - static_cast<uint8_t*>(base)) != goal_size) UTIL_THROW(FormatLoadException, "The data structures took " << (start - static_cast<uint8_t*>(base)) << " but Size says they should take " << goal_size);
}

namespace {
void ComplainAboutARPA(const Config &config, ModelType model_type) {
  if (config.write_mmap || !config.messages) return;
  if (config.arpa_complain == Config::ALL) {
    *config.messages << "Loading the LM will be faster if you build a binary file." << std::endl;
  } else if (config.arpa_complain == Config::EXPENSIVE &&
             (model_type == TRIE || model_type == QUANT_TRIE || model_type == ARRAY_TRIE || model_type == QUANT_ARRAY_TRIE)) {
    *config.messages << "Building " << kModelNames[model_type] << " from ARPA is expensive.  Save time by building a binary format." << std::endl;
  }
}

void CheckCounts(const std::vector<uint64_t> &counts) {
  UTIL_THROW_IF(counts.size() > KENLM_MAX_ORDER, FormatLoadException, "This model has order " << counts.size() << " but KenLM was compiled to support up to " << KENLM_MAX_ORDER << ".  " << KENLM_ORDER_MESSAGE);
  if (sizeof(uint64_t) > sizeof(std::size_t)) {
    for (std::vector<uint64_t>::const_iterator i = counts.begin(); i != counts.end(); ++i) {
      UTIL_THROW_IF(*i > static_cast<uint64_t>(std::numeric_limits<size_t>::max()), util::OverflowException, "This model has " << *i << " " << (i - counts.begin() + 1) << "-grams which is too many for 32-bit machines.");
    }
  }
}

} // namespace

template <class Search, class VocabularyT> GenericModel<Search, VocabularyT>::GenericModel(const char *file, const Config &init_config) : backing_(init_config) {
  util::scoped_fd fd(util::OpenReadOrThrow(file));
  if (IsBinaryFormat(fd.get())) {
    Parameters parameters;
    int fd_shallow = fd.release();
    backing_.InitializeBinary(fd_shallow, kModelType, kVersion, parameters);
    CheckCounts(parameters.counts);

    Config new_config(init_config);
    new_config.probing_multiplier = parameters.fixed.probing_multiplier;
    Search::UpdateConfigFromBinary(backing_, parameters.counts, VocabularyT::Size(parameters.counts[0], new_config), new_config);
    UTIL_THROW_IF(new_config.enumerate_vocab && !parameters.fixed.has_vocabulary, FormatLoadException, "The decoder requested all the vocabulary strings, but this binary file does not have them.  You may need to rebuild the binary file with an updated version of build_binary.");

    SetupMemory(backing_.LoadBinary(Size(parameters.counts, new_config)), parameters.counts, new_config);
    vocab_.LoadedBinary(parameters.fixed.has_vocabulary, fd_shallow, new_config.enumerate_vocab, backing_.VocabStringReadingOffset());
  } else {
    ComplainAboutARPA(init_config, kModelType);
    InitializeFromARPA(fd.release(), file, init_config);
  }

  // g++ prints warnings unless these are fully initialized.
  State begin_sentence = State();
  begin_sentence.length = 1;
  begin_sentence.words[0] = vocab_.BeginSentence();
  typename Search::Node ignored_node;
  bool ignored_independent_left;
  uint64_t ignored_extend_left;
  begin_sentence.backoff[0] = search_.LookupUnigram(begin_sentence.words[0], ignored_node, ignored_independent_left, ignored_extend_left).Backoff();
  State null_context = State();
  null_context.length = 0;
  P::Init(begin_sentence, null_context, vocab_, search_.Order());
}

template <class Search, class VocabularyT> void GenericModel<Search, VocabularyT>::InitializeFromARPA(int fd, const char *file, const Config &config) {
  // Backing file is the ARPA.
  util::FilePiece f(fd, file, config.ProgressMessages());
  try {
    std::vector<uint64_t> counts;
    // File counts do not include pruned trigrams that extend to quadgrams etc.   These will be fixed by search_.
    ReadARPACounts(f, counts);
    CheckCounts(counts);
    if (counts.size() < 2) UTIL_THROW(FormatLoadException, "This ngram implementation assumes at least a bigram model.");
    if (config.probing_multiplier <= 1.0) UTIL_THROW(ConfigException, "probing multiplier must be > 1.0");

    std::size_t vocab_size = util::CheckOverflow(VocabularyT::Size(counts[0], config));
    // Setup the binary file for writing the vocab lookup table.  The search_ is responsible for growing the binary file to its needs.
    vocab_.SetupMemory(backing_.SetupJustVocab(vocab_size, counts.size()), vocab_size, counts[0], config);

    if (config.write_mmap && config.include_vocab) {
      WriteWordsWrapper wrap(config.enumerate_vocab);
      vocab_.ConfigureEnumerate(&wrap, counts[0]);
      search_.InitializeFromARPA(file, f, counts, config, vocab_, backing_);
      void *vocab_rebase, *search_rebase;
      backing_.WriteVocabWords(wrap.Buffer(), vocab_rebase, search_rebase);
      // Due to writing at the end of file, mmap may have relocated data.  So remap.
      vocab_.Relocate(vocab_rebase);
      search_.SetupMemory(reinterpret_cast<uint8_t*>(search_rebase), counts, config);
    } else {
      vocab_.ConfigureEnumerate(config.enumerate_vocab, counts[0]);
      search_.InitializeFromARPA(file, f, counts, config, vocab_, backing_);
    }

    if (!vocab_.SawUnk()) {
      assert(config.unknown_missing != THROW_UP);
      // Default probabilities for unknown.
      search_.UnknownUnigram().backoff = 0.0;
      search_.UnknownUnigram().prob = config.unknown_missing_logprob;
    }
    backing_.FinishFile(config, kModelType, kVersion, counts);
  } catch (util::Exception &e) {
    e << " Byte: " << f.Offset();
    throw;
  }
}

template <class Search, class VocabularyT> FullScoreReturn GenericModel<Search, VocabularyT>::FullScore(const State &in_state, const WordIndex new_word, State &out_state) const {
  FullScoreReturn ret = ScoreExceptBackoff(in_state.words, in_state.words + in_state.length, new_word, out_state);
  for (const float *i = in_state.backoff + ret.ngram_length - 1; i < in_state.backoff + in_state.length; ++i) {
    ret.prob += *i;
  }
  return ret;
}

template <class Search, class VocabularyT> FullScoreReturn GenericModel<Search, VocabularyT>::FullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, State &out_state) const {
  context_rend = std::min(context_rend, context_rbegin + P::Order() - 1);
  FullScoreReturn ret = ScoreExceptBackoff(context_rbegin, context_rend, new_word, out_state);

  // Add the backoff weights for n-grams of order start to (context_rend - context_rbegin).
  unsigned char start = ret.ngram_length;
  if (context_rend - context_rbegin < static_cast<std::ptrdiff_t>(start)) return ret;

  bool independent_left;
  uint64_t extend_left;
  typename Search::Node node;
  if (start <= 1) {
    ret.prob += search_.LookupUnigram(*context_rbegin, node, independent_left, extend_left).Backoff();
    start = 2;
  } else if (!search_.FastMakeNode(context_rbegin, context_rbegin + start - 1, node)) {
    return ret;
  }
  // i is the order of the backoff we're looking for.
  unsigned char order_minus_2 = start - 2;
  for (const WordIndex *i = context_rbegin + start - 1; i < context_rend; ++i, ++order_minus_2) {
    typename Search::MiddlePointer p(search_.LookupMiddle(order_minus_2, *i, node, independent_left, extend_left));
    if (!p.Found()) break;
    ret.prob += p.Backoff();
  }
  return ret;
}

template <class Search, class VocabularyT> void GenericModel<Search, VocabularyT>::GetState(const WordIndex *context_rbegin, const WordIndex *context_rend, State &out_state) const {
  // Generate a state from context.
  context_rend = std::min(context_rend, context_rbegin + P::Order() - 1);
  if (context_rend == context_rbegin) {
    out_state.length = 0;
    return;
  }
  typename Search::Node node;
  bool independent_left;
  uint64_t extend_left;
  out_state.backoff[0] = search_.LookupUnigram(*context_rbegin, node, independent_left, extend_left).Backoff();
  out_state.length = HasExtension(out_state.backoff[0]) ? 1 : 0;
  float *backoff_out = out_state.backoff + 1;
  unsigned char order_minus_2 = 0;
  for (const WordIndex *i = context_rbegin + 1; i < context_rend; ++i, ++backoff_out, ++order_minus_2) {
    typename Search::MiddlePointer p(search_.LookupMiddle(order_minus_2, *i, node, independent_left, extend_left));
    if (!p.Found()) {
      std::copy(context_rbegin, context_rbegin + out_state.length, out_state.words);
      return;
    }
    *backoff_out = p.Backoff();
    if (HasExtension(*backoff_out)) out_state.length = i - context_rbegin + 1;
  }
  std::copy(context_rbegin, context_rbegin + out_state.length, out_state.words);
}

template <class Search, class VocabularyT> FullScoreReturn GenericModel<Search, VocabularyT>::ExtendLeft(
    const WordIndex *add_rbegin, const WordIndex *add_rend,
    const float *backoff_in,
    uint64_t extend_pointer,
    unsigned char extend_length,
    float *backoff_out,
    unsigned char &next_use) const {
  FullScoreReturn ret;
  typename Search::Node node;
  if (extend_length == 1) {
    typename Search::UnigramPointer ptr(search_.LookupUnigram(static_cast<WordIndex>(extend_pointer), node, ret.independent_left, ret.extend_left));
    ret.rest = ptr.Rest();
    ret.prob = ptr.Prob();
    assert(!ret.independent_left);
  } else {
    typename Search::MiddlePointer ptr(search_.Unpack(extend_pointer, extend_length, node));
    ret.rest = ptr.Rest();
    ret.prob = ptr.Prob();
    ret.extend_left = extend_pointer;
    // If this function is called, then it does depend on left words.
    ret.independent_left = false;
  }
  float subtract_me = ret.rest;
  ret.ngram_length = extend_length;
  next_use = extend_length;
  ResumeScore(add_rbegin, add_rend, extend_length - 1, node, backoff_out, next_use, ret);
  next_use -= extend_length;
  // Charge backoffs.
  for (const float *b = backoff_in + ret.ngram_length - extend_length; b < backoff_in + (add_rend - add_rbegin); ++b) ret.prob += *b;
  ret.prob -= subtract_me;
  ret.rest -= subtract_me;
  return ret;
}

template <class Search, class VocabularyT> uint64_t GenericModel<Search, VocabularyT>::GetEndOfSearchOffset() const {
  return backing_.VocabStringReadingOffset();
}

namespace {
// Do a paraonoid copy of history, assuming new_word has already been copied
// (hence the -1).  out_state.length could be zero so I avoided using
// std::copy.
void CopyRemainingHistory(const WordIndex *from, State &out_state) {
  WordIndex *out = out_state.words + 1;
  const WordIndex *in_end = from + static_cast<ptrdiff_t>(out_state.length) - 1;
  for (const WordIndex *in = from; in < in_end; ++in, ++out) *out = *in;
}
} // namespace

/* Ugly optimized function.  Produce a score excluding backoff.
 * The search goes in increasing order of ngram length.
 * Context goes backward, so context_begin is the word immediately preceeding
 * new_word.
 */
template <class Search, class VocabularyT> FullScoreReturn GenericModel<Search, VocabularyT>::ScoreExceptBackoff(
    const WordIndex *const context_rbegin,
    const WordIndex *const context_rend,
    const WordIndex new_word,
    State &out_state) const {
  assert(new_word < vocab_.Bound());
  FullScoreReturn ret;
  // ret.ngram_length contains the last known non-blank ngram length.
  ret.ngram_length = 1;

  typename Search::Node node;
  typename Search::UnigramPointer uni(search_.LookupUnigram(new_word, node, ret.independent_left, ret.extend_left));
  out_state.backoff[0] = uni.Backoff();
  ret.prob = uni.Prob();
  ret.rest = uni.Rest();

  // This is the length of the context that should be used for continuation to the right.
  out_state.length = HasExtension(out_state.backoff[0]) ? 1 : 0;
  // We'll write the word anyway since it will probably be used and does no harm being there.
  out_state.words[0] = new_word;
  if (context_rbegin == context_rend) return ret;

  ResumeScore(context_rbegin, context_rend, 0, node, out_state.backoff + 1, out_state.length, ret);
  CopyRemainingHistory(context_rbegin, out_state);
  return ret;
}

template <class Search, class VocabularyT> void GenericModel<Search, VocabularyT>::ResumeScore(const WordIndex *hist_iter, const WordIndex *const context_rend, unsigned char order_minus_2, typename Search::Node &node, float *backoff_out, unsigned char &next_use, FullScoreReturn &ret) const {
  for (; ; ++order_minus_2, ++hist_iter, ++backoff_out) {
    if (hist_iter == context_rend) return;
    if (ret.independent_left) return;
    if (order_minus_2 == P::Order() - 2) break;

    typename Search::MiddlePointer pointer(search_.LookupMiddle(order_minus_2, *hist_iter, node, ret.independent_left, ret.extend_left));
    if (!pointer.Found()) return;
    *backoff_out = pointer.Backoff();
    ret.prob = pointer.Prob();
    ret.rest = pointer.Rest();
    ret.ngram_length = order_minus_2 + 2;
    if (HasExtension(*backoff_out)) {
      next_use = ret.ngram_length;
    }
  }
  ret.independent_left = true;
  typename Search::LongestPointer longest(search_.LookupLongest(*hist_iter, node));
  if (longest.Found()) {
    ret.prob = longest.Prob();
    ret.rest = ret.prob;
    // There is no blank in longest_.
    ret.ngram_length = P::Order();
  }
}

template <class Search, class VocabularyT> float GenericModel<Search, VocabularyT>::InternalUnRest(const uint64_t *pointers_begin, const uint64_t *pointers_end, unsigned char first_length) const {
  float ret;
  typename Search::Node node;
  if (first_length == 1) {
    if (pointers_begin >= pointers_end) return 0.0;
    bool independent_left;
    uint64_t extend_left;
    typename Search::UnigramPointer ptr(search_.LookupUnigram(static_cast<WordIndex>(*pointers_begin), node, independent_left, extend_left));
    ret = ptr.Prob() - ptr.Rest();
    ++first_length;
    ++pointers_begin;
  } else {
    ret = 0.0;
  }
  for (const uint64_t *i = pointers_begin; i < pointers_end; ++i, ++first_length) {
    typename Search::MiddlePointer ptr(search_.Unpack(*i, first_length, node));
    ret += ptr.Prob() - ptr.Rest();
  }
  return ret;
}

template class GenericModel<HashedSearch<BackoffValue>, ProbingVocabulary>;
template class GenericModel<HashedSearch<RestValue>, ProbingVocabulary>;
template class GenericModel<trie::TrieSearch<DontQuantize, trie::DontBhiksha>, SortedVocabulary>;
template class GenericModel<trie::TrieSearch<DontQuantize, trie::ArrayBhiksha>, SortedVocabulary>;
template class GenericModel<trie::TrieSearch<SeparatelyQuantize, trie::DontBhiksha>, SortedVocabulary>;
template class GenericModel<trie::TrieSearch<SeparatelyQuantize, trie::ArrayBhiksha>, SortedVocabulary>;

} // namespace detail

base::Model *LoadVirtual(const char *file_name, const Config &config, ModelType model_type) {
  RecognizeBinary(file_name, model_type);
  switch (model_type) {
    case PROBING:
      return new ProbingModel(file_name, config);
    case REST_PROBING:
      return new RestProbingModel(file_name, config);
    case TRIE:
      return new TrieModel(file_name, config);
    case QUANT_TRIE:
      return new QuantTrieModel(file_name, config);
    case ARRAY_TRIE:
      return new ArrayTrieModel(file_name, config);
    case QUANT_ARRAY_TRIE:
      return new QuantArrayTrieModel(file_name, config);
    default:
      UTIL_THROW(FormatLoadException, "Confused by model type " << model_type);
  }
}

} // namespace ngram
} // namespace lm
