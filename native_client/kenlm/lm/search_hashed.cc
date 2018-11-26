#include "lm/search_hashed.hh"

#include "lm/binary_format.hh"
#include "lm/blank.hh"
#include "lm/lm_exception.hh"
#include "lm/model.hh"
#include "lm/read_arpa.hh"
#include "lm/value.hh"
#include "lm/vocab.hh"

#include "util/bit_packing.hh"
#include "util/file_piece.hh"

#include <string>

namespace lm {
namespace ngram {

class ProbingModel;

namespace {

/* These are passed to ReadNGrams so that n-grams with zero backoff that appear as context will still be used in state. */
template <class Middle> class ActivateLowerMiddle {
  public:
    explicit ActivateLowerMiddle(Middle &middle) : modify_(middle) {}

    void operator()(const WordIndex *vocab_ids, const unsigned int n) {
      uint64_t hash = static_cast<WordIndex>(vocab_ids[1]);
      for (const WordIndex *i = vocab_ids + 2; i < vocab_ids + n; ++i) {
        hash = detail::CombineWordHash(hash, *i);
      }
      typename Middle::MutableIterator i;
      // TODO: somehow get text of n-gram for this error message.
      if (!modify_.UnsafeMutableFind(hash, i))
        UTIL_THROW(FormatLoadException, "The context of every " << n << "-gram should appear as a " << (n-1) << "-gram");
      SetExtension(i->value.backoff);
    }

  private:
    Middle &modify_;
};

template <class Weights> class ActivateUnigram {
  public:
    explicit ActivateUnigram(Weights *unigram) : modify_(unigram) {}

    void operator()(const WordIndex *vocab_ids, const unsigned int /*n*/) {
      // assert(n == 2);
      SetExtension(modify_[vocab_ids[1]].backoff);
    }

  private:
    Weights *modify_;
};

// Find the lower order entry, inserting blanks along the way as necessary.
template <class Value> void FindLower(
    const std::vector<uint64_t> &keys,
    typename Value::Weights &unigram,
    std::vector<util::ProbingHashTable<typename Value::ProbingEntry, util::IdentityHash> > &middle,
    std::vector<typename Value::Weights *> &between) {
  typename util::ProbingHashTable<typename Value::ProbingEntry, util::IdentityHash>::MutableIterator iter;
  typename Value::ProbingEntry entry;
  // Backoff will always be 0.0.  We'll get the probability and rest in another pass.
  entry.value.backoff = kNoExtensionBackoff;
  // Go back and find the longest right-aligned entry, informing it that it extends left.  Normally this will match immediately, but sometimes SRI is dumb.
  for (int lower = keys.size() - 2; ; --lower) {
    if (lower == -1) {
      between.push_back(&unigram);
      return;
    }
    entry.key = keys[lower];
    bool found = middle[lower].FindOrInsert(entry, iter);
    between.push_back(&iter->value);
    if (found) return;
  }
}

// Between usually has  single entry, the value to adjust.  But sometimes SRI stupidly pruned entries so it has unitialized blank values to be set here.
template <class Added, class Build> void AdjustLower(
    const Added &added,
    const Build &build,
    std::vector<typename Build::Value::Weights *> &between,
    const unsigned int n,
    const std::vector<WordIndex> &vocab_ids,
    typename Build::Value::Weights *unigrams,
    std::vector<util::ProbingHashTable<typename Build::Value::ProbingEntry, util::IdentityHash> > &middle) {
  typedef typename Build::Value Value;
  if (between.size() == 1) {
    build.MarkExtends(*between.front(), added);
    return;
  }
  typedef util::ProbingHashTable<typename Value::ProbingEntry, util::IdentityHash> Middle;
  float prob = -fabs(between.back()->prob);
  // Order of the n-gram on which probabilities are based.
  unsigned char basis = n - between.size();
  assert(basis != 0);
  typename Build::Value::Weights **change = &between.back();
  // Skip the basis.
  --change;
  if (basis == 1) {
    // Hallucinate a bigram based on a unigram's backoff and a unigram probability.
    float &backoff = unigrams[vocab_ids[1]].backoff;
    SetExtension(backoff);
    prob += backoff;
    (*change)->prob = prob;
    build.SetRest(&*vocab_ids.begin(), 2, **change);
    basis = 2;
    --change;
  }
  uint64_t backoff_hash = static_cast<uint64_t>(vocab_ids[1]);
  for (unsigned char i = 2; i <= basis; ++i) {
    backoff_hash = detail::CombineWordHash(backoff_hash, vocab_ids[i]);
  }
  for (; basis < n - 1; ++basis, --change) {
    typename Middle::MutableIterator gotit;
    if (middle[basis - 2].UnsafeMutableFind(backoff_hash, gotit)) {
      float &backoff = gotit->value.backoff;
      SetExtension(backoff);
      prob += backoff;
    }
    (*change)->prob = prob;
    build.SetRest(&*vocab_ids.begin(), basis + 1, **change);
    backoff_hash = detail::CombineWordHash(backoff_hash, vocab_ids[basis+1]);
  }

  typename std::vector<typename Value::Weights *>::const_iterator i(between.begin());
  build.MarkExtends(**i, added);
  const typename Value::Weights *longer = *i;
  // Everything has probability but is not marked as extending.
  for (++i; i != between.end(); ++i) {
    build.MarkExtends(**i, *longer);
    longer = *i;
  }
}

// Continue marking lower entries even they know that they extend left.  This is used for upper/lower bounds.
template <class Build> void MarkLower(
    const std::vector<uint64_t> &keys,
    const Build &build,
    typename Build::Value::Weights &unigram,
    std::vector<util::ProbingHashTable<typename Build::Value::ProbingEntry, util::IdentityHash> > &middle,
    int start_order,
    const typename Build::Value::Weights &longer) {
  if (start_order == 0) return;
  // Hopefully the compiler will realize that if MarkExtends always returns false, it can simplify this code.
  for (int even_lower = start_order - 2 /* index in middle */; ; --even_lower) {
    if (even_lower == -1) {
      build.MarkExtends(unigram, longer);
      return;
    }
    if (!build.MarkExtends(
          middle[even_lower].UnsafeMutableMustFind(keys[even_lower])->value,
          longer)) return;
  }
}

template <class Build, class Activate, class Store> void ReadNGrams(
    util::FilePiece &f,
    const unsigned int n,
    const size_t count,
    const ProbingVocabulary &vocab,
    const Build &build,
    typename Build::Value::Weights *unigrams,
    std::vector<util::ProbingHashTable<typename Build::Value::ProbingEntry, util::IdentityHash> > &middle,
    Activate activate,
    Store &store,
    PositiveProbWarn &warn) {
  typedef typename Build::Value Value;
  assert(n >= 2);
  ReadNGramHeader(f, n);

  // Both vocab_ids and keys are non-empty because n >= 2.
  // vocab ids of words in reverse order.
  std::vector<WordIndex> vocab_ids(n);
  std::vector<uint64_t> keys(n-1);
  typename Store::Entry entry;
  std::vector<typename Value::Weights *> between;
  for (size_t i = 0; i < count; ++i) {
    ReadNGram(f, n, vocab, vocab_ids.rbegin(), entry.value, warn);
    build.SetRest(&*vocab_ids.begin(), n, entry.value);

    keys[0] = detail::CombineWordHash(static_cast<uint64_t>(vocab_ids.front()), vocab_ids[1]);
    for (unsigned int h = 1; h < n - 1; ++h) {
      keys[h] = detail::CombineWordHash(keys[h-1], vocab_ids[h+1]);
    }
    // Initially the sign bit is on, indicating it does not extend left.  Most already have this but there might +0.0.
    util::SetSign(entry.value.prob);
    entry.key = keys[n-2];

    store.Insert(entry);
    between.clear();
    FindLower<Value>(keys, unigrams[vocab_ids.front()], middle, between);
    AdjustLower<typename Store::Entry::Value, Build>(entry.value, build, between, n, vocab_ids, unigrams, middle);
    if (Build::kMarkEvenLower) MarkLower<Build>(keys, build, unigrams[vocab_ids.front()], middle, n - between.size() - 1, *between.back());
    activate(&*vocab_ids.begin(), n);
  }

  store.FinishedInserting();
}

} // namespace
namespace detail {

template <class Value> uint8_t *HashedSearch<Value>::SetupMemory(uint8_t *start, const std::vector<uint64_t> &counts, const Config &config) {
  unigram_ = Unigram(start, counts[0]);
  start += Unigram::Size(counts[0]);
  std::size_t allocated;
  middle_.clear();
  for (unsigned int n = 2; n < counts.size(); ++n) {
    allocated = Middle::Size(counts[n - 1], config.probing_multiplier);
    middle_.push_back(Middle(start, allocated));
    start += allocated;
  }
  allocated = Longest::Size(counts.back(), config.probing_multiplier);
  longest_ = Longest(start, allocated);
  start += allocated;
  return start;
}

/*template <class Value> void HashedSearch<Value>::Relocate(uint8_t *start, const std::vector<uint64_t> &counts, const Config &config) {
  unigram_ = Unigram(start, counts[0]);
  start += Unigram::Size(counts[0]);
  for (unsigned int n = 2; n < counts.size(); ++n) {
    middle[n-2].Relocate(start);
    start += Middle::Size(counts[n - 1], config.probing_multiplier)
  }
  longest_.Relocate(start);
}*/

template <class Value> void HashedSearch<Value>::InitializeFromARPA(const char * /*file*/, util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &config, ProbingVocabulary &vocab, BinaryFormat &backing) {
  void *vocab_rebase;
  void *search_base = backing.GrowForSearch(Size(counts, config), vocab.UnkCountChangePadding(), vocab_rebase);
  vocab.Relocate(vocab_rebase);
  SetupMemory(reinterpret_cast<uint8_t*>(search_base), counts, config);

  PositiveProbWarn warn(config.positive_log_probability);
  Read1Grams(f, counts[0], vocab, unigram_.Raw(), warn);
  CheckSpecials(config, vocab);
  DispatchBuild(f, counts, config, vocab, warn);
}

template <> void HashedSearch<BackoffValue>::DispatchBuild(util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &config, const ProbingVocabulary &vocab, PositiveProbWarn &warn) {
  NoRestBuild build;
  ApplyBuild(f, counts, vocab, warn, build);
}

template <> void HashedSearch<RestValue>::DispatchBuild(util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &config, const ProbingVocabulary &vocab, PositiveProbWarn &warn) {
  switch (config.rest_function) {
    case Config::REST_MAX:
      {
        MaxRestBuild build;
        ApplyBuild(f, counts, vocab, warn, build);
      }
      break;
    case Config::REST_LOWER:
      {
        LowerRestBuild<ProbingModel> build(config, counts.size(), vocab);
        ApplyBuild(f, counts, vocab, warn, build);
      }
      break;
  }
}

template <class Value> template <class Build> void HashedSearch<Value>::ApplyBuild(util::FilePiece &f, const std::vector<uint64_t> &counts, const ProbingVocabulary &vocab, PositiveProbWarn &warn, const Build &build) {
  for (WordIndex i = 0; i < counts[0]; ++i) {
    build.SetRest(&i, (unsigned int)1, unigram_.Raw()[i]);
  }

  try {
    if (counts.size() > 2) {
      ReadNGrams<Build, ActivateUnigram<typename Value::Weights>, Middle>(
          f, 2, counts[1], vocab, build, unigram_.Raw(), middle_, ActivateUnigram<typename Value::Weights>(unigram_.Raw()), middle_[0], warn);
    }
    for (unsigned int n = 3; n < counts.size(); ++n) {
      ReadNGrams<Build, ActivateLowerMiddle<Middle>, Middle>(
          f, n, counts[n-1], vocab, build, unigram_.Raw(), middle_, ActivateLowerMiddle<Middle>(middle_[n-3]), middle_[n-2], warn);
    }
    if (counts.size() > 2) {
      ReadNGrams<Build, ActivateLowerMiddle<Middle>, Longest>(
          f, counts.size(), counts[counts.size() - 1], vocab, build, unigram_.Raw(), middle_, ActivateLowerMiddle<Middle>(middle_.back()), longest_, warn);
    } else {
      ReadNGrams<Build, ActivateUnigram<typename Value::Weights>, Longest>(
          f, counts.size(), counts[counts.size() - 1], vocab, build, unigram_.Raw(), middle_, ActivateUnigram<typename Value::Weights>(unigram_.Raw()), longest_, warn);
    }
  } catch (util::ProbingSizeException &e) {
    UTIL_THROW(util::ProbingSizeException, "Avoid pruning n-grams like \"bar baz quux\" when \"foo bar baz quux\" is still in the model.  KenLM will work when this pruning happens, but the probing model assumes these events are rare enough that using blank space in the probing hash table will cover all of them.  Increase probing_multiplier (-p to build_binary) to add more blank spaces.\n");
  }
  ReadEnd(f);
}

template class HashedSearch<BackoffValue>;
template class HashedSearch<RestValue>;

} // namespace detail
} // namespace ngram
} // namespace lm
