#ifndef LM_TRIE_H
#define LM_TRIE_H

#include "lm/weights.hh"
#include "lm/word_index.hh"
#include "util/bit_packing.hh"

#include <cstddef>

#include <stdint.h>

namespace lm {
namespace ngram {
struct Config;
namespace trie {

struct NodeRange {
  uint64_t begin, end;
};

// TODO: if the number of unigrams is a concern, also bit pack these records.
struct UnigramValue {
  ProbBackoff weights;
  uint64_t next;
  uint64_t Next() const { return next; }
};

class UnigramPointer {
  public:
    explicit UnigramPointer(const ProbBackoff &to) : to_(&to) {}

    UnigramPointer() : to_(NULL) {}

    bool Found() const { return to_ != NULL; }

    float Prob() const { return to_->prob; }
    float Backoff() const { return to_->backoff; }
    float Rest() const { return Prob(); }

  private:
    const ProbBackoff *to_;
};

class Unigram {
  public:
    Unigram() {}

    void Init(void *start) {
      unigram_ = static_cast<UnigramValue*>(start);
    }

    static uint64_t Size(uint64_t count) {
      // +1 in case unknown doesn't appear.  +1 for the final next.
      return (count + 2) * sizeof(UnigramValue);
    }

    const ProbBackoff &Lookup(WordIndex index) const { return unigram_[index].weights; }

    ProbBackoff &Unknown() { return unigram_[0].weights; }

    UnigramValue *Raw() {
      return unigram_;
    }

    UnigramPointer Find(WordIndex word, NodeRange &next) const {
      UnigramValue *val = unigram_ + word;
      next.begin = val->next;
      next.end = (val+1)->next;
      return UnigramPointer(val->weights);
    }

  private:
    UnigramValue *unigram_;
};

class BitPacked {
  public:
    BitPacked() {}

    uint64_t InsertIndex() const {
      return insert_index_;
    }

  protected:
    static uint64_t BaseSize(uint64_t entries, uint64_t max_vocab, uint8_t remaining_bits);

    void BaseInit(void *base, uint64_t max_vocab, uint8_t remaining_bits);

    uint8_t word_bits_;
    uint8_t total_bits_;
    uint64_t word_mask_;

    uint8_t *base_;

    uint64_t insert_index_, max_vocab_;
};

template <class Bhiksha> class BitPackedMiddle : public BitPacked {
  public:
    static uint64_t Size(uint8_t quant_bits, uint64_t entries, uint64_t max_vocab, uint64_t max_next, const Config &config);

    // next_source need not be initialized.
    BitPackedMiddle(void *base, uint8_t quant_bits, uint64_t entries, uint64_t max_vocab, uint64_t max_next, const BitPacked &next_source, const Config &config);

    util::BitAddress Insert(WordIndex word);

    void FinishedLoading(uint64_t next_end, const Config &config);

    util::BitAddress Find(WordIndex word, NodeRange &range, uint64_t &pointer) const;

    util::BitAddress ReadEntry(uint64_t pointer, NodeRange &range) {
      uint64_t addr = pointer * total_bits_;
      addr += word_bits_;
      bhiksha_.ReadNext(base_, addr + quant_bits_, pointer, total_bits_, range);
      return util::BitAddress(base_, addr);
    }

  private:
    uint8_t quant_bits_;
    Bhiksha bhiksha_;

    const BitPacked *next_source_;
};

class BitPackedLongest : public BitPacked {
  public:
    static uint64_t Size(uint8_t quant_bits, uint64_t entries, uint64_t max_vocab) {
      return BaseSize(entries, max_vocab, quant_bits);
    }

    BitPackedLongest() {}

    void Init(void *base, uint8_t quant_bits, uint64_t max_vocab) {
      BaseInit(base, max_vocab, quant_bits);
    }

    util::BitAddress Insert(WordIndex word);

    util::BitAddress Find(WordIndex word, const NodeRange &node) const;
};

} // namespace trie
} // namespace ngram
} // namespace lm

#endif // LM_TRIE_H
