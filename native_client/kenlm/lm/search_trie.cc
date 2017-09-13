/* This is where the trie is built.  It's on-disk.  */
#include "lm/search_trie.hh"

#include "lm/bhiksha.hh"
#include "lm/binary_format.hh"
#include "lm/blank.hh"
#include "lm/lm_exception.hh"
#include "lm/max_order.hh"
#include "lm/quantize.hh"
#include "lm/trie.hh"
#include "lm/trie_sort.hh"
#include "lm/vocab.hh"
#include "lm/weights.hh"
#include "lm/word_index.hh"
#include "util/ersatz_progress.hh"
#include "util/mmap.hh"
#include "util/proxy_iterator.hh"
#include "util/scoped.hh"
#include "util/sized_iterator.hh"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <limits>
#include <numeric>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

namespace lm {
namespace ngram {
namespace trie {
namespace {

void ReadOrThrow(FILE *from, void *data, size_t size) {
  UTIL_THROW_IF(1 != std::fread(data, size, 1, from), util::ErrnoException, "Short read");
}

int Compare(unsigned char order, const void *first_void, const void *second_void) {
  const WordIndex *first = reinterpret_cast<const WordIndex*>(first_void), *second = reinterpret_cast<const WordIndex*>(second_void);
  const WordIndex *end = first + order;
  for (; first != end; ++first, ++second) {
    if (*first < *second) return -1;
    if (*first > *second) return 1;
  }
  return 0;
}

struct ProbPointer {
  unsigned char array;
  uint64_t index;
};

// Array of n-grams and float indices.
class BackoffMessages {
  public:
    void Init(std::size_t entry_size) {
      current_ = NULL;
      allocated_ = NULL;
      entry_size_ = entry_size;
    }

    void Add(const WordIndex *to, ProbPointer index) {
      while (current_ + entry_size_ > allocated_) {
        std::size_t allocated_size = allocated_ - (uint8_t*)backing_.get();
        Resize(std::max<std::size_t>(allocated_size * 2, entry_size_));
      }
      memcpy(current_, to, entry_size_ - sizeof(ProbPointer));
      *reinterpret_cast<ProbPointer*>(current_ + entry_size_ - sizeof(ProbPointer)) = index;
      current_ += entry_size_;
    }

    void Apply(float *const *const base, FILE *unigrams) {
      FinishedAdding();
      if (current_ == allocated_) return;
      rewind(unigrams);
      ProbBackoff weights;
      WordIndex unigram = 0;
      ReadOrThrow(unigrams, &weights, sizeof(weights));
      for (; current_ != allocated_; current_ += entry_size_) {
        const WordIndex &cur_word = *reinterpret_cast<const WordIndex*>(current_);
        for (; unigram < cur_word; ++unigram) {
          ReadOrThrow(unigrams, &weights, sizeof(weights));
        }
        if (!HasExtension(weights.backoff)) {
          weights.backoff = kExtensionBackoff;
          UTIL_THROW_IF(fseek(unigrams, -sizeof(weights), SEEK_CUR), util::ErrnoException, "Seeking backwards to denote unigram extension failed.");
          util::WriteOrThrow(unigrams, &weights, sizeof(weights));
        }
        const ProbPointer &write_to = *reinterpret_cast<const ProbPointer*>(current_ + sizeof(WordIndex));
        base[write_to.array][write_to.index] += weights.backoff;
      }
      backing_.reset();
    }

    void Apply(float *const *const base, RecordReader &reader) {
      FinishedAdding();
      if (current_ == allocated_) return;
      // We'll also use the same buffer to record messages to blanks that they extend.
      WordIndex *extend_out = reinterpret_cast<WordIndex*>(current_);
      const unsigned char order = (entry_size_ - sizeof(ProbPointer)) / sizeof(WordIndex);
      for (reader.Rewind(); reader && (current_ != allocated_); ) {
        switch (Compare(order, reader.Data(), current_)) {
          case -1:
            ++reader;
            break;
          case 1:
            // Message but nobody to receive it.  Write it down at the beginning of the buffer so we can inform this blank that it extends.
            for (const WordIndex *w = reinterpret_cast<const WordIndex *>(current_); w != reinterpret_cast<const WordIndex *>(current_) + order; ++w, ++extend_out) *extend_out = *w;
            current_ += entry_size_;
            break;
          case 0:
            float &backoff = reinterpret_cast<ProbBackoff*>((uint8_t*)reader.Data() + order * sizeof(WordIndex))->backoff;
            if (!HasExtension(backoff)) {
              backoff = kExtensionBackoff;
              reader.Overwrite(&backoff, sizeof(float));
            } else {
              const ProbPointer &write_to = *reinterpret_cast<const ProbPointer*>(current_ + entry_size_ - sizeof(ProbPointer));
              base[write_to.array][write_to.index] += backoff;
            }
            current_ += entry_size_;
            break;
        }
      }
      // Now this is a list of blanks that extend right.
      entry_size_ = sizeof(WordIndex) * order;
      Resize(sizeof(WordIndex) * (extend_out - (const WordIndex*)backing_.get()));
      current_ = (uint8_t*)backing_.get();
    }

    // Call after Apply
    bool Extends(unsigned char order, const WordIndex *words) {
      if (current_ == allocated_) return false;
      assert(order * sizeof(WordIndex) == entry_size_);
      while (true) {
        switch(Compare(order, words, current_)) {
          case 1:
            current_ += entry_size_;
            if (current_ == allocated_) return false;
            break;
          case -1:
            return false;
          case 0:
            return true;
        }
      }
    }

  private:
    void FinishedAdding() {
      Resize(current_ - (uint8_t*)backing_.get());
      // Sort requests in same order as files.
      util::SizedSort(backing_.get(), current_, entry_size_, EntryCompare((entry_size_ - sizeof(ProbPointer)) / sizeof(WordIndex)));
      current_ = (uint8_t*)backing_.get();
    }

    void Resize(std::size_t to) {
      std::size_t current = current_ - (uint8_t*)backing_.get();
      backing_.call_realloc(to);
      current_ = (uint8_t*)backing_.get() + current;
      allocated_ = (uint8_t*)backing_.get() + to;
    }

    util::scoped_malloc backing_;

    uint8_t *current_, *allocated_;

    std::size_t entry_size_;
};

const float kBadProb = std::numeric_limits<float>::infinity();

class SRISucks {
  public:
    SRISucks() {
      for (BackoffMessages *i = messages_; i != messages_ + KENLM_MAX_ORDER - 1; ++i)
        i->Init(sizeof(ProbPointer) + sizeof(WordIndex) * (i - messages_ + 1));
    }

    void Send(unsigned char begin, unsigned char order, const WordIndex *to, float prob_basis) {
      assert(prob_basis != kBadProb);
      ProbPointer pointer;
      pointer.array = order - 1;
      pointer.index = values_[order - 1].size();
      for (unsigned char i = begin; i < order; ++i) {
        messages_[i - 1].Add(to, pointer);
      }
      values_[order - 1].push_back(prob_basis);
    }

    void ObtainBackoffs(unsigned char total_order, FILE *unigram_file, RecordReader *reader) {
      for (unsigned char i = 0; i < KENLM_MAX_ORDER - 1; ++i) {
        it_[i] = values_[i].empty() ? NULL : &*values_[i].begin();
      }
      messages_[0].Apply(it_, unigram_file);
      BackoffMessages *messages = messages_ + 1;
      const RecordReader *end = reader + total_order - 2 /* exclude unigrams and longest order */;
      for (; reader != end; ++messages, ++reader) {
        messages->Apply(it_, *reader);
      }
    }

    ProbBackoff GetBlank(unsigned char total_order, unsigned char order, const WordIndex *indices) {
      assert(order > 1);
      ProbBackoff ret;
      ret.prob = *(it_[order - 1]++);
      ret.backoff = ((order != total_order - 1) && messages_[order - 1].Extends(order, indices)) ? kExtensionBackoff : kNoExtensionBackoff;
      return ret;
    }

    const std::vector<float> &Values(unsigned char order) const {
      return values_[order - 1];
    }

  private:
    // This used to be one array.  Then I needed to separate it by order for quantization to work.
    std::vector<float> values_[KENLM_MAX_ORDER - 1];
    BackoffMessages messages_[KENLM_MAX_ORDER - 1];

    float *it_[KENLM_MAX_ORDER - 1];
};

class FindBlanks {
  public:
    FindBlanks(unsigned char order, const ProbBackoff *unigrams, SRISucks &messages)
      : counts_(order), unigrams_(unigrams), sri_(messages) {}

    float UnigramProb(WordIndex index) const {
      return unigrams_[index].prob;
    }

    void Unigram(WordIndex /*index*/) {
      ++counts_[0];
    }

    void MiddleBlank(const unsigned char order, const WordIndex *indices, unsigned char lower, float prob_basis) {
      sri_.Send(lower, order, indices + 1, prob_basis);
      ++counts_[order - 1];
    }

    void Middle(const unsigned char order, const void * /*data*/) {
      ++counts_[order - 1];
    }

    void Longest(const void * /*data*/) {
      ++counts_.back();
    }

    const std::vector<uint64_t> &Counts() const {
      return counts_;
    }

  private:
    std::vector<uint64_t> counts_;

    const ProbBackoff *unigrams_;

    SRISucks &sri_;
};

// Phase to actually write n-grams to the trie.
template <class Quant, class Bhiksha> class WriteEntries {
  public:
    WriteEntries(RecordReader *contexts, const Quant &quant, UnigramValue *unigrams, BitPackedMiddle<Bhiksha> *middle, BitPackedLongest &longest, unsigned char order, SRISucks &sri) :
      contexts_(contexts),
      quant_(quant),
      unigrams_(unigrams),
      middle_(middle),
      longest_(longest),
      bigram_pack_((order == 2) ? static_cast<BitPacked&>(longest_) : static_cast<BitPacked&>(*middle_)),
      order_(order),
      sri_(sri) {}

    float UnigramProb(WordIndex index) const { return unigrams_[index].weights.prob; }

    void Unigram(WordIndex word) {
      unigrams_[word].next = bigram_pack_.InsertIndex();
    }

    void MiddleBlank(const unsigned char order, const WordIndex *indices, unsigned char /*lower*/, float /*prob_base*/) {
      ProbBackoff weights = sri_.GetBlank(order_, order, indices);
      typename Quant::MiddlePointer(quant_, order - 2, middle_[order - 2].Insert(indices[order - 1])).Write(weights.prob, weights.backoff);
    }

    void Middle(const unsigned char order, const void *data) {
      RecordReader &context = contexts_[order - 1];
      const WordIndex *words = reinterpret_cast<const WordIndex*>(data);
      ProbBackoff weights = *reinterpret_cast<const ProbBackoff*>(words + order);
      if (context && !memcmp(data, context.Data(), sizeof(WordIndex) * order)) {
        SetExtension(weights.backoff);
        ++context;
      }
      typename Quant::MiddlePointer(quant_, order - 2, middle_[order - 2].Insert(words[order - 1])).Write(weights.prob, weights.backoff);
    }

    void Longest(const void *data) {
      const WordIndex *words = reinterpret_cast<const WordIndex*>(data);
      typename Quant::LongestPointer(quant_, longest_.Insert(words[order_ - 1])).Write(reinterpret_cast<const Prob*>(words + order_)->prob);
    }

  private:
    RecordReader *contexts_;
    const Quant &quant_;
    UnigramValue *const unigrams_;
    BitPackedMiddle<Bhiksha> *const middle_;
    BitPackedLongest &longest_;
    BitPacked &bigram_pack_;
    const unsigned char order_;
    SRISucks &sri_;
};

struct Gram {
  Gram(const WordIndex *in_begin, unsigned char order) : begin(in_begin), end(in_begin + order) {}

  const WordIndex *begin, *end;

  // For queue, this is the direction we want.
  bool operator<(const Gram &other) const {
    return std::lexicographical_compare(other.begin, other.end, begin, end);
  }
};

template <class Doing> class BlankManager {
  public:
    BlankManager(unsigned char total_order, Doing &doing) : total_order_(total_order), been_length_(0), doing_(doing) {
      for (float *i = basis_; i != basis_ + KENLM_MAX_ORDER - 1; ++i) *i = kBadProb;
    }

    void Visit(const WordIndex *to, unsigned char length, float prob) {
      basis_[length - 1] = prob;
      unsigned char overlap = std::min<unsigned char>(length - 1, been_length_);
      const WordIndex *cur;
      WordIndex *pre;
      for (cur = to, pre = been_; cur != to + overlap; ++cur, ++pre) {
        if (*pre != *cur) break;
      }
      if (cur == to + length - 1) {
        *pre = *cur;
        been_length_ = length;
        return;
      }
      // There are blanks to insert starting with order blank.
      unsigned char blank = cur - to + 1;
      UTIL_THROW_IF(blank == 1, FormatLoadException, "Missing a unigram that appears as context.");
      const float *lower_basis;
      for (lower_basis = basis_ + blank - 2; *lower_basis == kBadProb; --lower_basis) {}
      unsigned char based_on = lower_basis - basis_ + 1;
      for (; cur != to + length - 1; ++blank, ++cur, ++pre) {
        assert(*lower_basis != kBadProb);
        doing_.MiddleBlank(blank, to, based_on, *lower_basis);
        *pre = *cur;
        // Mark that the probability is a blank so it shouldn't be used as the basis for a later n-gram.
        basis_[blank - 1] = kBadProb;
      }
      *pre = *cur;
      been_length_ = length;
    }

  private:
    const unsigned char total_order_;

    WordIndex been_[KENLM_MAX_ORDER];
    unsigned char been_length_;

    float basis_[KENLM_MAX_ORDER];

    Doing &doing_;
};

template <class Doing> void RecursiveInsert(const unsigned char total_order, const WordIndex unigram_count, RecordReader *input, std::ostream *progress_out, const char *message, Doing &doing) {
  util::ErsatzProgress progress(unigram_count + 1, progress_out, message);
  WordIndex unigram = 0;
  std::priority_queue<Gram> grams;
  if (unigram_count) grams.push(Gram(&unigram, 1));
  for (unsigned char i = 2; i <= total_order; ++i) {
    if (input[i-2]) grams.push(Gram(reinterpret_cast<const WordIndex*>(input[i-2].Data()), i));
  }

  BlankManager<Doing> blank(total_order, doing);

  while (!grams.empty()) {
    Gram top = grams.top();
    grams.pop();
    unsigned char order = top.end - top.begin;
    if (order == 1) {
      blank.Visit(&unigram, 1, doing.UnigramProb(unigram));
      doing.Unigram(unigram);
      progress.Set(unigram);
      if (++unigram < unigram_count) grams.push(top);
    } else {
      if (order == total_order) {
        blank.Visit(top.begin, order, reinterpret_cast<const Prob*>(top.end)->prob);
        doing.Longest(top.begin);
      } else {
        blank.Visit(top.begin, order, reinterpret_cast<const ProbBackoff*>(top.end)->prob);
        doing.Middle(order, top.begin);
      }
      RecordReader &reader = input[order - 2];
      if (++reader) grams.push(top);
    }
  }
}

void SanityCheckCounts(const std::vector<uint64_t> &initial, const std::vector<uint64_t> &fixed) {
  if (fixed[0] != initial[0]) UTIL_THROW(util::Exception, "Unigram count should be constant but initial is " << initial[0] << " and recounted is " << fixed[0]);
  if (fixed.back() != initial.back()) UTIL_THROW(util::Exception, "Longest count should be constant but it changed from " << initial.back() << " to " << fixed.back());
  for (unsigned char i = 0; i < initial.size(); ++i) {
    if (fixed[i] < initial[i]) UTIL_THROW(util::Exception, "Counts came out lower than expected.  This shouldn't happen");
  }
}

template <class Quant> void TrainQuantizer(uint8_t order, uint64_t count, const std::vector<float> &additional, RecordReader &reader, util::ErsatzProgress &progress, Quant &quant) {
  std::vector<float> probs(additional), backoffs;
  probs.reserve(count + additional.size());
  backoffs.reserve(count);
  for (reader.Rewind(); reader; ++reader) {
    const ProbBackoff &weights = *reinterpret_cast<const ProbBackoff*>(reinterpret_cast<const uint8_t*>(reader.Data()) + sizeof(WordIndex) * order);
    probs.push_back(weights.prob);
    if (weights.backoff != 0.0) backoffs.push_back(weights.backoff);
    ++progress;
  }
  quant.Train(order, probs, backoffs);
}

template <class Quant> void TrainProbQuantizer(uint8_t order, uint64_t count, RecordReader &reader, util::ErsatzProgress &progress, Quant &quant) {
  std::vector<float> probs, backoffs;
  probs.reserve(count);
  for (reader.Rewind(); reader; ++reader) {
    const Prob &weights = *reinterpret_cast<const Prob*>(reinterpret_cast<const uint8_t*>(reader.Data()) + sizeof(WordIndex) * order);
    probs.push_back(weights.prob);
    ++progress;
  }
  quant.TrainProb(order, probs);
}

void PopulateUnigramWeights(FILE *file, WordIndex unigram_count, RecordReader &contexts, UnigramValue *unigrams) {
  // Fill unigram probabilities.
  try {
    rewind(file);
    for (WordIndex i = 0; i < unigram_count; ++i) {
      ReadOrThrow(file, &unigrams[i].weights, sizeof(ProbBackoff));
      if (contexts && *reinterpret_cast<const WordIndex*>(contexts.Data()) == i) {
        SetExtension(unigrams[i].weights.backoff);
        ++contexts;
      }
    }
  } catch (util::Exception &e) {
    e << " while re-reading unigram probabilities";
    throw;
  }
}

} // namespace

template <class Quant, class Bhiksha> void BuildTrie(SortedFiles &files, std::vector<uint64_t> &counts, const Config &config, TrieSearch<Quant, Bhiksha> &out, Quant &quant, SortedVocabulary &vocab, BinaryFormat &backing) {
  RecordReader inputs[KENLM_MAX_ORDER - 1];
  RecordReader contexts[KENLM_MAX_ORDER - 1];

  for (unsigned char i = 2; i <= counts.size(); ++i) {
    inputs[i-2].Init(files.Full(i), i * sizeof(WordIndex) + (i == counts.size() ? sizeof(Prob) : sizeof(ProbBackoff)));
    contexts[i-2].Init(files.Context(i), (i-1) * sizeof(WordIndex));
  }

  SRISucks sri;
  std::vector<uint64_t> fixed_counts;
  util::scoped_FILE unigram_file;
  util::scoped_fd unigram_fd(files.StealUnigram());
  {
    util::scoped_memory unigrams;
    MapRead(util::POPULATE_OR_READ, unigram_fd.get(), 0, counts[0] * sizeof(ProbBackoff), unigrams);
    FindBlanks finder(counts.size(), reinterpret_cast<const ProbBackoff*>(unigrams.get()), sri);
    RecursiveInsert(counts.size(), counts[0], inputs, config.ProgressMessages(), "Identifying n-grams omitted by SRI", finder);
    fixed_counts = finder.Counts();
  }
  unigram_file.reset(util::FDOpenOrThrow(unigram_fd));
  for (const RecordReader *i = inputs; i != inputs + counts.size() - 2; ++i) {
    if (*i) UTIL_THROW(FormatLoadException, "There's a bug in the trie implementation: the " << (i - inputs + 2) << "-gram table did not complete reading");
  }
  SanityCheckCounts(counts, fixed_counts);
  counts = fixed_counts;

  sri.ObtainBackoffs(counts.size(), unigram_file.get(), inputs);

  void *vocab_relocate;
  void *search_base = backing.GrowForSearch(TrieSearch<Quant, Bhiksha>::Size(fixed_counts, config), vocab.UnkCountChangePadding(), vocab_relocate);
  vocab.Relocate(vocab_relocate);
  out.SetupMemory(reinterpret_cast<uint8_t*>(search_base), fixed_counts, config);

  for (unsigned char i = 2; i <= counts.size(); ++i) {
    inputs[i-2].Rewind();
  }
  if (Quant::kTrain) {
    util::ErsatzProgress progress(std::accumulate(counts.begin() + 1, counts.end(), 0),
                                  config.ProgressMessages(), "Quantizing");
    for (unsigned char i = 2; i < counts.size(); ++i) {
      TrainQuantizer(i, counts[i-1], sri.Values(i), inputs[i-2], progress, quant);
    }
    TrainProbQuantizer(counts.size(), counts.back(), inputs[counts.size() - 2], progress, quant);
    quant.FinishedLoading(config);
  }

  UnigramValue *unigrams = out.unigram_.Raw();
  PopulateUnigramWeights(unigram_file.get(), counts[0], contexts[0], unigrams);
  unigram_file.reset();

  for (unsigned char i = 2; i <= counts.size(); ++i) {
    inputs[i-2].Rewind();
  }
  // Fill entries except unigram probabilities.
  {
    WriteEntries<Quant, Bhiksha> writer(contexts, quant, unigrams, out.middle_begin_, out.longest_, counts.size(), sri);
    RecursiveInsert(counts.size(), counts[0], inputs, config.ProgressMessages(), "Writing trie", writer);
    // Write the last unigram entry, which is the end pointer for the bigrams.
    writer.Unigram(counts[0]);
  }

  // Do not disable this error message or else too little state will be returned.  Both WriteEntries::Middle and returning state based on found n-grams will need to be fixed to handle this situation.
  for (unsigned char order = 2; order <= counts.size(); ++order) {
    const RecordReader &context = contexts[order - 2];
    if (context) {
      FormatLoadException e;
      e << "A " << static_cast<unsigned int>(order) << "-gram has context";
      const WordIndex *ctx = reinterpret_cast<const WordIndex*>(context.Data());
      for (const WordIndex *i = ctx; i != ctx + order - 1; ++i) {
        e << ' ' << *i;
      }
      e << " so this context must appear in the model as a " << static_cast<unsigned int>(order - 1) << "-gram but it does not";
      throw e;
    }
  }

  /* Set ending offsets so the last entry will be sized properly */
  // Last entry for unigrams was already set.
  if (out.middle_begin_ != out.middle_end_) {
    for (typename TrieSearch<Quant, Bhiksha>::Middle *i = out.middle_begin_; i != out.middle_end_ - 1; ++i) {
      i->FinishedLoading((i+1)->InsertIndex(), config);
    }
    (out.middle_end_ - 1)->FinishedLoading(out.longest_.InsertIndex(), config);
  }
}

template <class Quant, class Bhiksha> uint8_t *TrieSearch<Quant, Bhiksha>::SetupMemory(uint8_t *start, const std::vector<uint64_t> &counts, const Config &config) {
  quant_.SetupMemory(start, counts.size(), config);
  start += Quant::Size(counts.size(), config);
  unigram_.Init(start);
  start += Unigram::Size(counts[0]);
  FreeMiddles();
  middle_begin_ = static_cast<Middle*>(malloc(sizeof(Middle) * (counts.size() - 2)));
  middle_end_ = middle_begin_ + (counts.size() - 2);
  std::vector<uint8_t*> middle_starts(counts.size() - 2);
  for (unsigned char i = 2; i < counts.size(); ++i) {
    middle_starts[i-2] = start;
    start += Middle::Size(Quant::MiddleBits(config), counts[i-1], counts[0], counts[i], config);
  }
  // Crazy backwards thing so we initialize using pointers to ones that have already been initialized
  for (unsigned char i = counts.size() - 1; i >= 2; --i) {
    // use "placement new" syntax to initalize Middle in an already-allocated memory location
    new (middle_begin_ + i - 2) Middle(
        middle_starts[i-2],
        quant_.MiddleBits(config),
        counts[i-1],
        counts[0],
        counts[i],
        (i == counts.size() - 1) ? static_cast<const BitPacked&>(longest_) : static_cast<const BitPacked &>(middle_begin_[i-1]),
        config);
  }
  longest_.Init(start, quant_.LongestBits(config), counts[0]);
  return start + Longest::Size(Quant::LongestBits(config), counts.back(), counts[0]);
}

template <class Quant, class Bhiksha> void TrieSearch<Quant, Bhiksha>::InitializeFromARPA(const char *file, util::FilePiece &f, std::vector<uint64_t> &counts, const Config &config, SortedVocabulary &vocab, BinaryFormat &backing) {
  std::string temporary_prefix;
  if (!config.temporary_directory_prefix.empty()) {
    temporary_prefix = config.temporary_directory_prefix;
  } else if (config.write_mmap) {
    temporary_prefix = config.write_mmap;
  } else {
    temporary_prefix = file;
  }
  // At least 1MB sorting memory.
  SortedFiles sorted(config, f, counts, std::max<size_t>(config.building_memory, 1048576), temporary_prefix, vocab);

  BuildTrie(sorted, counts, config, *this, quant_, vocab, backing);
}

template class TrieSearch<DontQuantize, DontBhiksha>;
template class TrieSearch<DontQuantize, ArrayBhiksha>;
template class TrieSearch<SeparatelyQuantize, DontBhiksha>;
template class TrieSearch<SeparatelyQuantize, ArrayBhiksha>;

} // namespace trie
} // namespace ngram
} // namespace lm
