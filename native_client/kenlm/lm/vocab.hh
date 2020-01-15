#ifndef LM_VOCAB_H
#define LM_VOCAB_H

#include "lm/enumerate_vocab.hh"
#include "lm/lm_exception.hh"
#include "lm/virtual_interface.hh"
#include "util/file_stream.hh"
#include "util/murmur_hash.hh"
#include "util/pool.hh"
#include "util/probing_hash_table.hh"
#include "util/sorted_uniform.hh"
#include "util/string_piece.hh"

#include <limits>
#include <string>
#include <vector>

namespace lm {
struct ProbBackoff;
class EnumerateVocab;

namespace ngram {
struct Config;

namespace detail {
uint64_t HashForVocab(const char *str, std::size_t len);
inline uint64_t HashForVocab(const StringPiece &str) {
  return HashForVocab(str.data(), str.length());
}
struct ProbingVocabularyHeader;
} // namespace detail

// Writes words immediately to a file instead of buffering, because we know
// where in the file to put them.
class ImmediateWriteWordsWrapper : public EnumerateVocab {
  public:
    ImmediateWriteWordsWrapper(EnumerateVocab *inner, int fd, uint64_t start);

    void Add(WordIndex index, const StringPiece &str) {
      stream_ << str << '\0';
      if (inner_) inner_->Add(index, str);
    }

  private:
    EnumerateVocab *inner_;

    util::FileStream stream_;
};

// When the binary size isn't known yet.
class WriteWordsWrapper : public EnumerateVocab {
  public:
    WriteWordsWrapper(EnumerateVocab *inner);

    void Add(WordIndex index, const StringPiece &str);

    const std::string &Buffer() const { return buffer_; }
    void Write(int fd, uint64_t start);

  private:
    EnumerateVocab *inner_;

    std::string buffer_;
};

// Vocabulary based on sorted uniform find storing only uint64_t values and using their offsets as indices.
class SortedVocabulary : public base::Vocabulary {
  public:
    SortedVocabulary();

    WordIndex Index(const StringPiece &str) const {
      const uint64_t *found;
      if (util::BoundedSortedUniformFind<const uint64_t*, util::IdentityAccessor<uint64_t>, util::Pivot64>(
            util::IdentityAccessor<uint64_t>(),
            begin_ - 1, 0,
            end_, std::numeric_limits<uint64_t>::max(),
            detail::HashForVocab(str), found)) {
        return found - begin_ + 1; // +1 because <unk> is 0 and does not appear in the lookup table.
      } else {
        return 0;
      }
    }

    // Size for purposes of file writing
    static uint64_t Size(uint64_t entries, const Config &config);

    /* Read null-delimited words from file from_words, renumber according to
     * hash order, write null-delimited words to to_words, and create a mapping
     * from old id to new id.  The 0th vocab word must be <unk>.
     */
    static void ComputeRenumbering(WordIndex types, int from_words, int to_words, std::vector<WordIndex> &mapping);

    // Vocab words are [0, Bound())  Only valid after FinishedLoading/LoadedBinary.
    WordIndex Bound() const { return bound_; }

    // Everything else is for populating.  I'm too lazy to hide and friend these, but you'll only get a const reference anyway.
    void SetupMemory(void *start, std::size_t allocated, std::size_t entries, const Config &config);

    void Relocate(void *new_start);

    void ConfigureEnumerate(EnumerateVocab *to, std::size_t max_entries);

    // Insert and FinishedLoading go together.
    WordIndex Insert(const StringPiece &str);
    // Reorders reorder_vocab so that the IDs are sorted.
    void FinishedLoading(ProbBackoff *reorder_vocab);

    // Trie stores the correct counts including <unk> in the header.  If this was previously sized based on a count exluding <unk>, padding with 8 bytes will make it the correct size based on a count including <unk>.
    std::size_t UnkCountChangePadding() const { return SawUnk() ? 0 : sizeof(uint64_t); }

    bool SawUnk() const { return saw_unk_; }

    void LoadedBinary(bool have_words, int fd, EnumerateVocab *to, uint64_t offset);

    uint64_t *&EndHack() { return end_; }

    void Populated();

  private:
    template <class T> void GenericFinished(T *reorder);

    uint64_t *begin_, *end_;

    WordIndex bound_;

    bool saw_unk_;

    EnumerateVocab *enumerate_;

    // Actual strings.  Used only when loading from ARPA and enumerate_ != NULL
    util::Pool string_backing_;

    std::vector<StringPiece> strings_to_enumerate_;
};

#pragma pack(push)
#pragma pack(4)
struct ProbingVocabularyEntry {
  uint64_t key;
  WordIndex value;

  typedef uint64_t Key;
  uint64_t GetKey() const { return key; }
  void SetKey(uint64_t to) { key = to; }

  static ProbingVocabularyEntry Make(uint64_t key, WordIndex value) {
    ProbingVocabularyEntry ret;
    ret.key = key;
    ret.value = value;
    return ret;
  }
};
#pragma pack(pop)

// Vocabulary storing a map from uint64_t to WordIndex.
class ProbingVocabulary : public base::Vocabulary {
  public:
    ProbingVocabulary();

    WordIndex Index(const StringPiece &str) const {
      Lookup::ConstIterator i;
      return lookup_.Find(detail::HashForVocab(str), i) ? i->value : 0;
    }

    static uint64_t Size(uint64_t entries, float probing_multiplier);
    // This just unwraps Config to get the probing_multiplier.
    static uint64_t Size(uint64_t entries, const Config &config);

    // Vocab words are [0, Bound()).
    WordIndex Bound() const { return bound_; }

    // Everything else is for populating.  I'm too lazy to hide and friend these, but you'll only get a const reference anyway.
    void SetupMemory(void *start, std::size_t allocated);
    void SetupMemory(void *start, std::size_t allocated, std::size_t /*entries*/, const Config &/*config*/) {
      SetupMemory(start, allocated);
    }

    void Relocate(void *new_start);

    void ConfigureEnumerate(EnumerateVocab *to, std::size_t max_entries);

    WordIndex Insert(const StringPiece &str);

    template <class Weights> void FinishedLoading(Weights * /*reorder_vocab*/) {
      InternalFinishedLoading();
    }

    std::size_t UnkCountChangePadding() const { return 0; }

    bool SawUnk() const { return saw_unk_; }

    void LoadedBinary(bool have_words, int fd, EnumerateVocab *to, uint64_t offset);

  private:
    void InternalFinishedLoading();

    typedef util::ProbingHashTable<ProbingVocabularyEntry, util::IdentityHash> Lookup;

    Lookup lookup_;

    WordIndex bound_;

    bool saw_unk_;

    EnumerateVocab *enumerate_;

    detail::ProbingVocabularyHeader *header_;
};

void MissingUnknown(const Config &config);
void MissingSentenceMarker(const Config &config, const char *str);

template <class Vocab> void CheckSpecials(const Config &config, const Vocab &vocab) {
  if (!vocab.SawUnk()) MissingUnknown(config);
  if (vocab.BeginSentence() == vocab.NotFound()) MissingSentenceMarker(config, "<s>");
  if (vocab.EndSentence() == vocab.NotFound()) MissingSentenceMarker(config, "</s>");
}

class WriteUniqueWords {
  public:
    explicit WriteUniqueWords(int fd) : word_list_(fd) {}

    void operator()(const StringPiece &word) {
      word_list_ << word << '\0';
    }

  private:
    util::FileStream word_list_;
};

class NoOpUniqueWords {
  public:
    NoOpUniqueWords() {}
    void operator()(const StringPiece &word) {}
};

template <class NewWordAction = NoOpUniqueWords> class GrowableVocab {
  public:
    static std::size_t MemUsage(WordIndex content) {
      return Lookup::MemUsage(content > 2 ? content : 2);
    }

    // Does not take ownership of new_word_construct
    template <class NewWordConstruct> GrowableVocab(WordIndex initial_size, const NewWordConstruct &new_word_construct = NewWordAction())
      : lookup_(initial_size), new_word_(new_word_construct) {
      FindOrInsert("<unk>"); // Force 0
      FindOrInsert("<s>"); // Force 1
      FindOrInsert("</s>"); // Force 2
    }

    WordIndex Index(const StringPiece &str) const {
      Lookup::ConstIterator i;
      return lookup_.Find(detail::HashForVocab(str), i) ? i->value : 0;
    }

    WordIndex FindOrInsert(const StringPiece &word) {
      ProbingVocabularyEntry entry = ProbingVocabularyEntry::Make(util::MurmurHashNative(word.data(), word.size()), Size());
      Lookup::MutableIterator it;
      if (!lookup_.FindOrInsert(entry, it)) {
        new_word_(word);
        UTIL_THROW_IF(Size() >= std::numeric_limits<lm::WordIndex>::max(), VocabLoadException, "Too many vocabulary words.  Change WordIndex to uint64_t in lm/word_index.hh");
      }
      return it->value;
    }

    WordIndex Size() const { return lookup_.Size(); }

    bool IsSpecial(WordIndex word) const {
      return word <= 2;
    }

  private:
    typedef util::AutoProbing<ProbingVocabularyEntry, util::IdentityHash> Lookup;

    Lookup lookup_;

    NewWordAction new_word_;
};

} // namespace ngram
} // namespace lm

#endif // LM_VOCAB_H
