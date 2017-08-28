#ifndef LM_VIRTUAL_INTERFACE_H
#define LM_VIRTUAL_INTERFACE_H

#include "lm/return.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

#include <string>
#include <cstring>

namespace lm {
namespace base {

template <class T, class U, class V> class ModelFacade;

/* Vocabulary interface.  Call Index(string) and get a word index for use in
 * calling Model.  It provides faster convenience functions for <s>, </s>, and
 * <unk> although you can also find these using Index.
 *
 * Some models do not load the mapping from index to string.  If you need this,
 * check if the model Vocabulary class implements such a function and access it
 * directly.
 *
 * The Vocabulary object is always owned by the Model and can be retrieved from
 * the Model using BaseVocabulary() for this abstract interface or
 * GetVocabulary() for the actual implementation (in which case you'll need the
 * actual implementation of the Model too).
 */
class Vocabulary {
  public:
    virtual ~Vocabulary();

    WordIndex BeginSentence() const { return begin_sentence_; }
    WordIndex EndSentence() const { return end_sentence_; }
    WordIndex NotFound() const { return not_found_; }

    /* Most implementations allow StringPiece lookups and need only override
     * Index(StringPiece).  SRI requires null termination and overrides all
     * three methods.
     */
    virtual WordIndex Index(const StringPiece &str) const = 0;
    virtual WordIndex Index(const std::string &str) const {
      return Index(StringPiece(str));
    }
    virtual WordIndex Index(const char *str) const {
      return Index(StringPiece(str));
    }

  protected:
    // Call SetSpecial afterward.
    Vocabulary() {}

    Vocabulary(WordIndex begin_sentence, WordIndex end_sentence, WordIndex not_found) {
      SetSpecial(begin_sentence, end_sentence, not_found);
    }

    void SetSpecial(WordIndex begin_sentence, WordIndex end_sentence, WordIndex not_found);

    WordIndex begin_sentence_, end_sentence_, not_found_;

  private:
    // Disable copy constructors.  They're private and undefined.
    // Ersatz boost::noncopyable.
    Vocabulary(const Vocabulary &);
    Vocabulary &operator=(const Vocabulary &);
};

/* There are two ways to access a Model.
 *
 *
 * OPTION 1: Access the Model directly (e.g. lm::ngram::Model in model.hh).
 *
 * Every Model implements the scoring function:
 * float Score(
 *   const Model::State &in_state,
 *   const WordIndex new_word,
 *   Model::State &out_state) const;
 *
 * It can also return the length of n-gram matched by the model:
 * FullScoreReturn FullScore(
 *   const Model::State &in_state,
 *   const WordIndex new_word,
 *   Model::State &out_state) const;
 *
 *
 * There are also accessor functions:
 * const State &BeginSentenceState() const;
 * const State &NullContextState() const;
 * const Vocabulary &GetVocabulary() const;
 * unsigned int Order() const;
 *
 * NB: In case you're wondering why the model implementation looks like it's
 * missing these methods, see facade.hh.
 *
 * This is the fastest way to use a model and presents a normal State class to
 * be included in a hypothesis state structure.
 *
 *
 * OPTION 2: Use the virtual interface below.
 *
 * The virtual interface allow you to decide which Model to use at runtime
 * without templatizing everything on the Model type.  However, each Model has
 * its own State class, so a single State cannot be efficiently provided (it
 * would require using the maximum memory of any Model's State or memory
 * allocation with each lookup).  This means you become responsible for
 * allocating memory with size StateSize() and passing it to the Score or
 * FullScore functions provided here.
 *
 * For example, cdec has a std::string containing the entire state of a
 * hypothesis.  It can reserve StateSize bytes in this string for the model
 * state.
 *
 * All the State objects are POD, so it's ok to use raw memory for storing
 * State.
 * in_state and out_state must not have the same address.
 */
class Model {
  public:
    virtual ~Model();

    size_t StateSize() const { return state_size_; }
    const void *BeginSentenceMemory() const { return begin_sentence_memory_; }
    void BeginSentenceWrite(void *to) const { memcpy(to, begin_sentence_memory_, StateSize()); }
    const void *NullContextMemory() const { return null_context_memory_; }
    void NullContextWrite(void *to) const { memcpy(to, null_context_memory_, StateSize()); }

    // Requires in_state != out_state
    virtual float BaseScore(const void *in_state, const WordIndex new_word, void *out_state) const = 0;

    // Requires in_state != out_state
    virtual FullScoreReturn BaseFullScore(const void *in_state, const WordIndex new_word, void *out_state) const = 0;

    // Prefer to use FullScore.  The context words should be provided in reverse order.
    virtual FullScoreReturn BaseFullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, void *out_state) const = 0;

    unsigned char Order() const { return order_; }

    const Vocabulary &BaseVocabulary() const { return *base_vocab_; }

  private:
    template <class T, class U, class V> friend class ModelFacade;
    explicit Model(size_t state_size) : state_size_(state_size) {}

    const size_t state_size_;
    const void *begin_sentence_memory_, *null_context_memory_;

    const Vocabulary *base_vocab_;

    unsigned char order_;

    // Disable copy constructors.  They're private and undefined.
    // Ersatz boost::noncopyable.
    Model(const Model &);
    Model &operator=(const Model &);
};

} // mamespace base
} // namespace lm

#endif // LM_VIRTUAL_INTERFACE_H
