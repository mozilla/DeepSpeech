#ifndef LM_SEARCH_TRIE_H
#define LM_SEARCH_TRIE_H

#include "lm/config.hh"
#include "lm/model_type.hh"
#include "lm/return.hh"
#include "lm/trie.hh"
#include "lm/weights.hh"

#include "util/file.hh"
#include "util/file_piece.hh"

#include <vector>
#include <cstdlib>
#include <cassert>

namespace lm {
namespace ngram {
class BinaryFormat;
class SortedVocabulary;
namespace trie {

template <class Quant, class Bhiksha> class TrieSearch;
class SortedFiles;
template <class Quant, class Bhiksha> void BuildTrie(SortedFiles &files, std::vector<uint64_t> &counts, const Config &config, TrieSearch<Quant, Bhiksha> &out, Quant &quant, SortedVocabulary &vocab, BinaryFormat &backing);

template <class Quant, class Bhiksha> class TrieSearch {
  public:
    typedef NodeRange Node;

    typedef ::lm::ngram::trie::UnigramPointer UnigramPointer;
    typedef typename Quant::MiddlePointer MiddlePointer;
    typedef typename Quant::LongestPointer LongestPointer;

    static const bool kDifferentRest = false;

    static const ModelType kModelType = static_cast<ModelType>(TRIE_SORTED + Quant::kModelTypeAdd + Bhiksha::kModelTypeAdd);

    static const unsigned int kVersion = 1;

    static void UpdateConfigFromBinary(const BinaryFormat &file, const std::vector<uint64_t> &counts, uint64_t offset, Config &config) {
      Quant::UpdateConfigFromBinary(file, offset, config);
      // Currently the unigram pointers are not compresssed, so there will only be a header for order > 2.
      if (counts.size() > 2)
        Bhiksha::UpdateConfigFromBinary(file, offset + Quant::Size(counts.size(), config) + Unigram::Size(counts[0]), config);
    }

    static uint64_t Size(const std::vector<uint64_t> &counts, const Config &config) {
      uint64_t ret = Quant::Size(counts.size(), config) + Unigram::Size(counts[0]);
      for (unsigned char i = 1; i < counts.size() - 1; ++i) {
        ret += Middle::Size(Quant::MiddleBits(config), counts[i], counts[0], counts[i+1], config);
      }
      return ret + Longest::Size(Quant::LongestBits(config), counts.back(), counts[0]);
    }

    TrieSearch() : middle_begin_(NULL), middle_end_(NULL) {}

    ~TrieSearch() { FreeMiddles(); }

    uint8_t *SetupMemory(uint8_t *start, const std::vector<uint64_t> &counts, const Config &config);

    void InitializeFromARPA(const char *file, util::FilePiece &f, std::vector<uint64_t> &counts, const Config &config, SortedVocabulary &vocab, BinaryFormat &backing);

    unsigned char Order() const {
      return middle_end_ - middle_begin_ + 2;
    }

    ProbBackoff &UnknownUnigram() { return unigram_.Unknown(); }

    UnigramPointer LookupUnigram(WordIndex word, Node &next, bool &independent_left, uint64_t &extend_left) const {
      extend_left = static_cast<uint64_t>(word);
      UnigramPointer ret(unigram_.Find(word, next));
      independent_left = (next.begin == next.end);
      return ret;
    }

    MiddlePointer Unpack(uint64_t extend_pointer, unsigned char extend_length, Node &node) const {
      return MiddlePointer(quant_, extend_length - 2, middle_begin_[extend_length - 2].ReadEntry(extend_pointer, node));
    }

    MiddlePointer LookupMiddle(unsigned char order_minus_2, WordIndex word, Node &node, bool &independent_left, uint64_t &extend_left) const {
      util::BitAddress address(middle_begin_[order_minus_2].Find(word, node, extend_left));
      independent_left = (address.base == NULL) || (node.begin == node.end);
      return MiddlePointer(quant_, order_minus_2, address);
    }

    LongestPointer LookupLongest(WordIndex word, const Node &node) const {
      return LongestPointer(quant_, longest_.Find(word, node));
    }

    bool FastMakeNode(const WordIndex *begin, const WordIndex *end, Node &node) const {
      assert(begin != end);
      bool independent_left;
      uint64_t ignored;
      LookupUnigram(*begin, node, independent_left, ignored);
      for (const WordIndex *i = begin + 1; i < end; ++i) {
        if (independent_left || !LookupMiddle(i - begin - 1, *i, node, independent_left, ignored).Found()) return false;
      }
      return true;
    }

  private:
    friend void BuildTrie<Quant, Bhiksha>(SortedFiles &files, std::vector<uint64_t> &counts, const Config &config, TrieSearch<Quant, Bhiksha> &out, Quant &quant, SortedVocabulary &vocab, BinaryFormat &backing);

    // Middles are managed manually so we can delay construction and they don't have to be copyable.
    void FreeMiddles() {
      for (const Middle *i = middle_begin_; i != middle_end_; ++i) {
        i->~Middle();
      }
      std::free(middle_begin_);
    }

    typedef trie::BitPackedMiddle<Bhiksha> Middle;

    typedef trie::BitPackedLongest Longest;
    Longest longest_;

    Middle *middle_begin_, *middle_end_;
    Quant quant_;

    typedef ::lm::ngram::trie::Unigram Unigram;
    Unigram unigram_;
};

} // namespace trie
} // namespace ngram
} // namespace lm

#endif // LM_SEARCH_TRIE_H
