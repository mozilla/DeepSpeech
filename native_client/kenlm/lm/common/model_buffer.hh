#ifndef LM_COMMON_MODEL_BUFFER_H
#define LM_COMMON_MODEL_BUFFER_H

/* Format with separate files in suffix order.  Each file contains
 * n-grams of the same order.
 */
#include "lm/word_index.hh"
#include "util/file.hh"
#include "util/fixed_array.hh"
#include "util/string_piece.hh"

#include <string>
#include <vector>

namespace util { namespace stream {
class Chains;
class Chain;
}} // namespaces

namespace lm {

namespace ngram { class State; }

class ModelBuffer {
  public:
    // Construct for writing.  Must call VocabFile() and fill it with null-delimited vocab words.
    ModelBuffer(StringPiece file_base, bool keep_buffer, bool output_q);

    // Load from file.
    explicit ModelBuffer(StringPiece file_base);

    // Must call VocabFile and populate before calling this function.
    void Sink(util::stream::Chains &chains, const std::vector<uint64_t> &counts);

    // Read files and write to the given chains.  If fewer chains are provided,
    // only do the lower orders.
    void Source(util::stream::Chains &chains);

    void Source(std::size_t order_minus_1, util::stream::Chain &chain);

    // The order of the n-gram model that is associated with the model buffer.
    std::size_t Order() const { return counts_.size(); }
    // Requires Sink or load from file.
    const std::vector<uint64_t> &Counts() const {
      assert(!counts_.empty());
      return counts_;
    }

    int VocabFile() const { return vocab_file_.get(); }

    int RawFile(std::size_t order_minus_1) const {
      return files_[order_minus_1].get();
    }

    bool Keep() const { return keep_buffer_; }

    // Slowly execute a language model query with binary search.
    // This is used by interpolation to gather tuning probabilities rather than
    // scanning the files.
    float SlowQuery(const ngram::State &context, WordIndex word, ngram::State &out) const;

  private:
    const std::string file_base_;
    const bool keep_buffer_;
    bool output_q_;
    std::vector<uint64_t> counts_;

    util::scoped_fd vocab_file_;
    util::FixedArray<util::scoped_fd> files_;
};

} // namespace lm

#endif // LM_COMMON_MODEL_BUFFER_H
