#ifndef LM_COMMON_NGRAM_H
#define LM_COMMON_NGRAM_H

#include "lm/weights.hh"
#include "lm/word_index.hh"

#include <cstddef>
#include <cassert>
#include <stdint.h>
#include <cstring>

namespace lm {

class NGramHeader {
  public:
    NGramHeader(void *begin, std::size_t order)
      : begin_(static_cast<WordIndex*>(begin)), end_(begin_ + order) {}

    NGramHeader() : begin_(NULL), end_(NULL) {}

    const uint8_t *Base() const { return reinterpret_cast<const uint8_t*>(begin_); }
    uint8_t *Base() { return reinterpret_cast<uint8_t*>(begin_); }

    void ReBase(void *to) {
      std::size_t difference = end_ - begin_;
      begin_ = reinterpret_cast<WordIndex*>(to);
      end_ = begin_ + difference;
    }

    // These are for the vocab index.
    // Lower-case in deference to STL.
    const WordIndex *begin() const { return begin_; }
    WordIndex *begin() { return begin_; }
    const WordIndex *end() const { return end_; }
    WordIndex *end() { return end_; }

    std::size_t size() const { return end_ - begin_; }
    std::size_t Order() const { return end_ - begin_; }

  private:
    WordIndex *begin_, *end_;
};

template <class PayloadT> class NGram : public NGramHeader {
  public:
    typedef PayloadT Payload;

    NGram() : NGramHeader(NULL, 0) {}

    NGram(void *begin, std::size_t order) : NGramHeader(begin, order) {}

    // Would do operator++ but that can get confusing for a stream.
    void NextInMemory() {
      ReBase(&Value() + 1);
    }

    static std::size_t TotalSize(std::size_t order) {
      return order * sizeof(WordIndex) + sizeof(Payload);
    }
    std::size_t TotalSize() const {
      // Compiler should optimize this.
      return TotalSize(Order());
    }

    static std::size_t OrderFromSize(std::size_t size) {
      std::size_t ret = (size - sizeof(Payload)) / sizeof(WordIndex);
      assert(size == TotalSize(ret));
      return ret;
    }

    const Payload &Value() const { return *reinterpret_cast<const Payload *>(end()); }
    Payload &Value() { return *reinterpret_cast<Payload *>(end()); }
};

} // namespace lm

#endif // LM_COMMON_NGRAM_H
