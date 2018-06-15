#ifndef LM_BINARY_FORMAT_H
#define LM_BINARY_FORMAT_H

#include "lm/config.hh"
#include "lm/model_type.hh"
#include "lm/read_arpa.hh"

#include "util/file_piece.hh"
#include "util/mmap.hh"
#include "util/scoped.hh"

#include <cstddef>
#include <vector>

#include <stdint.h>

namespace lm {
namespace ngram {

extern const char *kModelNames[6];

/*Inspect a file to determine if it is a binary lm.  If not, return false.
 * If so, return true and set recognized to the type.  This is the only API in
 * this header designed for use by decoder authors.
 */
bool RecognizeBinary(const char *file, ModelType &recognized);

struct FixedWidthParameters {
  unsigned char order;
  float probing_multiplier;
  // What type of model is this?
  ModelType model_type;
  // Does the end of the file have the actual strings in the vocabulary?
  bool has_vocabulary;
  unsigned int search_version;
};

// This is a macro instead of an inline function so constants can be assigned using it.
#define ALIGN8(a) ((std::ptrdiff_t(((a)-1)/8)+1)*8)

// Parameters stored in the header of a binary file.
struct Parameters {
  FixedWidthParameters fixed;
  std::vector<uint64_t> counts;
};

class BinaryFormat {
  public:
    explicit BinaryFormat(const Config &config);

    // Reading a binary file:
    // Takes ownership of fd
    void InitializeBinary(int fd, ModelType model_type, unsigned int search_version, Parameters &params);
    // Used to read parts of the file to update the config object before figuring out full size.
    void ReadForConfig(void *to, std::size_t amount, uint64_t offset_excluding_header) const;
    // Actually load the binary file and return a pointer to the beginning of the search area.
    void *LoadBinary(std::size_t size);

    uint64_t VocabStringReadingOffset() const {
      assert(vocab_string_offset_ != kInvalidOffset);
      return vocab_string_offset_;
    }

    // Writing a binary file or initializing in RAM from ARPA:
    // Size for vocabulary.
    void *SetupJustVocab(std::size_t memory_size, uint8_t order);
    // Warning: can change the vocaulary base pointer.
    void *GrowForSearch(std::size_t memory_size, std::size_t vocab_pad, void *&vocab_base);
    // Warning: can change vocabulary and search base addresses.
    void WriteVocabWords(const std::string &buffer, void *&vocab_base, void *&search_base);
    // Write the header at the beginning of the file.
    void FinishFile(const Config &config, ModelType model_type, unsigned int search_version, const std::vector<uint64_t> &counts);

  private:
    void MapFile(void *&vocab_base, void *&search_base);

    // Copied from configuration.
    const Config::WriteMethod write_method_;
    const char *write_mmap_;
    util::LoadMethod load_method_;

    // File behind memory, if any.
    util::scoped_fd file_;

    // If there is a file involved, a single mapping.
    util::scoped_memory mapping_;

    // If the data is only in memory, separately allocate each because the trie
    // knows vocab's size before it knows search's size (because SRILM might
    // have pruned).
    util::scoped_memory memory_vocab_, memory_search_;

    // Memory ranges.  Note that these may not be contiguous and may not all
    // exist.
    std::size_t header_size_, vocab_size_, vocab_pad_;
    // aka end of search.
    uint64_t vocab_string_offset_;

    static const uint64_t kInvalidOffset = (uint64_t)-1;
};

bool IsBinaryFormat(int fd);

} // namespace ngram
} // namespace lm
#endif // LM_BINARY_FORMAT_H
