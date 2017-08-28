#ifndef LM_COMMON_PRINT_H
#define LM_COMMON_PRINT_H

#include "lm/word_index.hh"
#include "util/mmap.hh"
#include "util/string_piece.hh"

#include <cassert>
#include <vector>

namespace util { namespace stream { class ChainPositions; }}

// Warning: PrintARPA routines read all unigrams before all bigrams before all
// trigrams etc.  So if other parts of the chain move jointly, you'll have to
// buffer.

namespace lm {

class VocabReconstitute {
  public:
    // fd must be alive for life of this object; does not take ownership.
    explicit VocabReconstitute(int fd);

    const char *Lookup(WordIndex index) const {
      assert(index < map_.size() - 1);
      return map_[index];
    }

    StringPiece LookupPiece(WordIndex index) const {
      return StringPiece(map_[index], map_[index + 1] - 1 - map_[index]);
    }

    std::size_t Size() const {
      // There's an extra entry to support StringPiece lengths.
      return map_.size() - 1;
    }

  private:
    util::scoped_memory memory_;
    std::vector<const char*> map_;
};

class PrintARPA {
  public:
    // Does not take ownership of vocab_fd or out_fd.
    explicit PrintARPA(int vocab_fd, int out_fd, const std::vector<uint64_t> &counts)
      : vocab_fd_(vocab_fd), out_fd_(out_fd), counts_(counts) {}

    void Run(const util::stream::ChainPositions &positions);

  private:
    int vocab_fd_;
    int out_fd_;
    std::vector<uint64_t> counts_;
};

} // namespace lm
#endif // LM_COMMON_PRINT_H
