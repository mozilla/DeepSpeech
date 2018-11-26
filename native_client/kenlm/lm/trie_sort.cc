#include "lm/trie_sort.hh"

#include "lm/config.hh"
#include "lm/lm_exception.hh"
#include "lm/read_arpa.hh"
#include "lm/vocab.hh"
#include "lm/weights.hh"
#include "lm/word_index.hh"
#include "util/file_piece.hh"
#include "util/mmap.hh"
#include "util/pool.hh"
#include "util/proxy_iterator.hh"
#include "util/sized_iterator.hh"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <limits>
#include <vector>

namespace lm {
namespace ngram {
namespace trie {
namespace {

typedef util::SizedIterator NGramIter;

// Proxy for an entry except there is some extra cruft between the entries.  This is used to sort (n-1)-grams using the same memory as the sorted n-grams.
class PartialViewProxy {
  public:
    PartialViewProxy() : attention_size_(0), inner_() {}

    PartialViewProxy(void *ptr, std::size_t block_size, util::FreePool &pool) : attention_size_(pool.ElementSize()), inner_(ptr, block_size), pool_(&pool) {}

    operator util::ValueBlock() const {
      return util::ValueBlock(inner_.Data(), *pool_);
    }

    PartialViewProxy &operator=(const PartialViewProxy &from) {
      memcpy(inner_.Data(), from.inner_.Data(), attention_size_);
      return *this;
    }

    PartialViewProxy &operator=(const util::ValueBlock &from) {
      memcpy(inner_.Data(), from.Data(), attention_size_);
      return *this;
    }

    const void *Data() const { return inner_.Data(); }
    void *Data() { return inner_.Data(); }

    friend void swap(PartialViewProxy first, PartialViewProxy second);

  private:
    friend class util::ProxyIterator<PartialViewProxy>;

    typedef util::ValueBlock value_type;

    const std::size_t attention_size_;

    typedef util::SizedInnerIterator InnerIterator;
    InnerIterator &Inner() { return inner_; }
    const InnerIterator &Inner() const { return inner_; }
    InnerIterator inner_;

    util::FreePool *pool_;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
void swap(PartialViewProxy first, PartialViewProxy second) {
  std::swap_ranges(reinterpret_cast<char*>(first.Data()), reinterpret_cast<char*>(first.Data()) + first.attention_size_, reinterpret_cast<char*>(second.Data()));
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

typedef util::ProxyIterator<PartialViewProxy> PartialIter;

FILE *DiskFlush(const void *mem_begin, const void *mem_end, const std::string &temp_prefix) {
  util::scoped_fd file(util::MakeTemp(temp_prefix));
  util::WriteOrThrow(file.get(), mem_begin, (uint8_t*)mem_end - (uint8_t*)mem_begin);
  return util::FDOpenOrThrow(file);
}

FILE *WriteContextFile(uint8_t *begin, uint8_t *end, const std::string &temp_prefix, std::size_t entry_size, unsigned char order) {
  const size_t context_size = sizeof(WordIndex) * (order - 1);

  util::FreePool pool(context_size);
  // Sort just the contexts using the same memory.
  PartialIter context_begin(PartialViewProxy(begin + sizeof(WordIndex), entry_size, pool));
  PartialIter context_end(PartialViewProxy(end + sizeof(WordIndex), entry_size, pool));

#if defined(_WIN32) || defined(_WIN64)
  std::stable_sort
#else
  std::sort
#endif
    (context_begin, context_end, util::SizedCompare<EntryCompare, PartialViewProxy>(EntryCompare(order - 1)));

  util::scoped_FILE out(util::FMakeTemp(temp_prefix));

  // Write out to file and uniqueify at the same time.  Could have used unique_copy if there was an appropriate OutputIterator.
  if (context_begin == context_end) return out.release();
  PartialIter i(context_begin);
  util::WriteOrThrow(out.get(), i->Data(), context_size);
  const void *previous = i->Data();
  ++i;
  for (; i != context_end; ++i) {
    if (memcmp(previous, i->Data(), context_size)) {
      util::WriteOrThrow(out.get(), i->Data(), context_size);
      previous = i->Data();
    }
  }
  return out.release();
}

struct ThrowCombine {
  void operator()(std::size_t entry_size, unsigned char order, const void *first, const void *second, FILE * /*out*/) const {
    const WordIndex *base = reinterpret_cast<const WordIndex*>(first);
    FormatLoadException e;
    e << "Duplicate n-gram detected with vocab ids";
    for (const WordIndex *i = base; i != base + order; ++i) {
      e << ' ' << *i;
    }
    throw e;
  }
};

// Useful for context files that just contain records with no value.
struct FirstCombine {
  void operator()(std::size_t entry_size, unsigned char /*order*/, const void *first, const void * /*second*/, FILE *out) const {
    util::WriteOrThrow(out, first, entry_size);
  }
};

template <class Combine> FILE *MergeSortedFiles(FILE *first_file, FILE *second_file, const std::string &temp_prefix, std::size_t weights_size, unsigned char order, const Combine &combine) {
  std::size_t entry_size = sizeof(WordIndex) * order + weights_size;
  RecordReader first, second;
  first.Init(first_file, entry_size);
  second.Init(second_file, entry_size);
  util::scoped_FILE out_file(util::FMakeTemp(temp_prefix));
  EntryCompare less(order);
  while (first && second) {
    if (less(first.Data(), second.Data())) {
      util::WriteOrThrow(out_file.get(), first.Data(), entry_size);
      ++first;
    } else if (less(second.Data(), first.Data())) {
      util::WriteOrThrow(out_file.get(), second.Data(), entry_size);
      ++second;
    } else {
      combine(entry_size, order, first.Data(), second.Data(), out_file.get());
      ++first; ++second;
    }
  }
  for (RecordReader &remains = (first ? first : second); remains; ++remains) {
    util::WriteOrThrow(out_file.get(), remains.Data(), entry_size);
  }
  return out_file.release();
}

} // namespace

void RecordReader::Init(FILE *file, std::size_t entry_size) {
  entry_size_ = entry_size;
  data_.reset(malloc(entry_size));
  UTIL_THROW_IF(!data_.get(), util::ErrnoException, "Failed to malloc read buffer");
  file_ = file;
  if (file) {
    rewind(file);
    remains_ = true;
    ++*this;
  } else {
    remains_ = false;
  }
}

void RecordReader::Overwrite(const void *start, std::size_t amount) {
  long internal = (uint8_t*)start - (uint8_t*)data_.get();
  UTIL_THROW_IF(fseek(file_, internal - entry_size_, SEEK_CUR), util::ErrnoException, "Couldn't seek backwards for revision");
  util::WriteOrThrow(file_, start, amount);
  long forward = entry_size_ - internal - amount;
#if !defined(_WIN32) && !defined(_WIN64)
  if (forward)
#endif
    UTIL_THROW_IF(fseek(file_, forward, SEEK_CUR), util::ErrnoException, "Couldn't seek forwards past revision");
}

void RecordReader::Rewind() {
  if (file_) {
    rewind(file_);
    remains_ = true;
    ++*this;
  } else {
    remains_ = false;
  }
}

SortedFiles::SortedFiles(const Config &config, util::FilePiece &f, std::vector<uint64_t> &counts, size_t buffer, const std::string &file_prefix, SortedVocabulary &vocab) {
  PositiveProbWarn warn(config.positive_log_probability);
  unigram_.reset(util::MakeTemp(file_prefix));
  {
    // In case <unk> appears.
    size_t size_out = (counts[0] + 1) * sizeof(ProbBackoff);
    util::scoped_mmap unigram_mmap(util::MapZeroedWrite(unigram_.get(), size_out), size_out);
    Read1Grams(f, counts[0], vocab, reinterpret_cast<ProbBackoff*>(unigram_mmap.get()), warn);
    CheckSpecials(config, vocab);
    if (!vocab.SawUnk()) ++counts[0];
  }

  // Only use as much buffer as we need.
  size_t buffer_use = 0;
  for (unsigned int order = 2; order < counts.size(); ++order) {
    buffer_use = std::max<size_t>(buffer_use, static_cast<size_t>((sizeof(WordIndex) * order + 2 * sizeof(float)) * counts[order - 1]));
  }
  buffer_use = std::max<size_t>(buffer_use, static_cast<size_t>((sizeof(WordIndex) * counts.size() + sizeof(float)) * counts.back()));
  buffer = std::min<size_t>(buffer, buffer_use);

  util::scoped_malloc mem;
  mem.reset(malloc(buffer));
  if (!mem.get()) UTIL_THROW(util::ErrnoException, "malloc failed for sort buffer size " << buffer);

  for (unsigned char order = 2; order <= counts.size(); ++order) {
    ConvertToSorted(f, vocab, counts, file_prefix, order, warn, mem.get(), buffer);
  }
  ReadEnd(f);
}

namespace {
class Closer {
  public:
    explicit Closer(std::deque<FILE*> &files) : files_(files) {}

    ~Closer() {
      for (std::deque<FILE*>::iterator i = files_.begin(); i != files_.end(); ++i) {
        util::scoped_FILE deleter(*i);
      }
    }

    void PopFront() {
      util::scoped_FILE deleter(files_.front());
      files_.pop_front();
    }
  private:
    std::deque<FILE*> &files_;
};
} // namespace

void SortedFiles::ConvertToSorted(util::FilePiece &f, const SortedVocabulary &vocab, const std::vector<uint64_t> &counts, const std::string &file_prefix, unsigned char order, PositiveProbWarn &warn, void *mem, std::size_t mem_size) {
  ReadNGramHeader(f, order);
  const size_t count = counts[order - 1];
  // Size of weights.  Does it include backoff?
  const size_t words_size = sizeof(WordIndex) * order;
  const size_t weights_size = sizeof(float) + ((order == counts.size()) ? 0 : sizeof(float));
  const size_t entry_size = words_size + weights_size;
  const size_t batch_size = std::min(count, mem_size / entry_size);
  uint8_t *const begin = reinterpret_cast<uint8_t*>(mem);

  std::deque<FILE*> files, contexts;
  Closer files_closer(files), contexts_closer(contexts);

  for (std::size_t batch = 0, done = 0; done < count; ++batch) {
    uint8_t *out = begin;
    uint8_t *out_end = out + std::min(count - done, batch_size) * entry_size;
    if (order == counts.size()) {
      for (; out != out_end; out += entry_size) {
        std::reverse_iterator<WordIndex*> it(reinterpret_cast<WordIndex*>(out) + order);
        ReadNGram(f, order, vocab, it, *reinterpret_cast<Prob*>(out + words_size), warn);
      }
    } else {
      for (; out != out_end; out += entry_size) {
        std::reverse_iterator<WordIndex*> it(reinterpret_cast<WordIndex*>(out) + order);
        ReadNGram(f, order, vocab, it, *reinterpret_cast<ProbBackoff*>(out + words_size), warn);
      }
    }
    // Sort full records by full n-gram.
    util::SizedSort(begin, out_end, entry_size, EntryCompare(order));
    files.push_back(DiskFlush(begin, out_end, file_prefix));
    contexts.push_back(WriteContextFile(begin, out_end, file_prefix, entry_size, order));

    done += (out_end - begin) / entry_size;
  }

  // All individual files created.  Merge them.

  while (files.size() > 1) {
    files.push_back(MergeSortedFiles(files[0], files[1], file_prefix, weights_size, order, ThrowCombine()));
    files_closer.PopFront();
    files_closer.PopFront();
    contexts.push_back(MergeSortedFiles(contexts[0], contexts[1], file_prefix, 0, order - 1, FirstCombine()));
    contexts_closer.PopFront();
    contexts_closer.PopFront();
  }

  if (!files.empty()) {
    // Steal from closers.
    full_[order - 2].reset(files.front());
    files.pop_front();
    context_[order - 2].reset(contexts.front());
    contexts.pop_front();
  }
}

} // namespace trie
} // namespace ngram
} // namespace lm
