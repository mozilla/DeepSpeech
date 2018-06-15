#ifndef LM_FACADE_H
#define LM_FACADE_H

#include "lm/virtual_interface.hh"
#include "util/string_piece.hh"

#include <string>

namespace lm {
namespace base {

// Common model interface that depends on knowing the specific classes.
// Curiously recurring template pattern.
template <class Child, class StateT, class VocabularyT> class ModelFacade : public Model {
  public:
    typedef StateT State;
    typedef VocabularyT Vocabulary;

    /* Translate from void* to State */
    FullScoreReturn BaseFullScore(const void *in_state, const WordIndex new_word, void *out_state) const {
      return static_cast<const Child*>(this)->FullScore(
          *reinterpret_cast<const State*>(in_state),
          new_word,
          *reinterpret_cast<State*>(out_state));
    }

    FullScoreReturn BaseFullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, void *out_state) const {
      return static_cast<const Child*>(this)->FullScoreForgotState(
          context_rbegin,
          context_rend,
          new_word,
          *reinterpret_cast<State*>(out_state));
    }

    // Default Score function calls FullScore.  Model can override this.
    float Score(const State &in_state, const WordIndex new_word, State &out_state) const {
      return static_cast<const Child*>(this)->FullScore(in_state, new_word, out_state).prob;
    }

    float BaseScore(const void *in_state, const WordIndex new_word, void *out_state) const {
      return static_cast<const Child*>(this)->Score(
          *reinterpret_cast<const State*>(in_state),
          new_word,
          *reinterpret_cast<State*>(out_state));
    }

    const State &BeginSentenceState() const { return begin_sentence_; }
    const State &NullContextState() const { return null_context_; }
    const Vocabulary &GetVocabulary() const { return *static_cast<const Vocabulary*>(&BaseVocabulary()); }

  protected:
    ModelFacade() : Model(sizeof(State)) {}

    virtual ~ModelFacade() {}

    // begin_sentence and null_context can disappear after.  vocab should stay.
    void Init(const State &begin_sentence, const State &null_context, const Vocabulary &vocab, unsigned char order) {
      begin_sentence_ = begin_sentence;
      null_context_ = null_context;
      begin_sentence_memory_ = &begin_sentence_;
      null_context_memory_ = &null_context_;
      base_vocab_ = &vocab;
      order_ = order;
    }

  private:
    State begin_sentence_, null_context_;
};

} // mamespace base
} // namespace lm

#endif // LM_FACADE_H
