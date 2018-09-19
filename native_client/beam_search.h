#ifndef BEAM_SEARCH_H
#define BEAM_SEARCH_H

#include "alphabet.h"
#include "trie_node.h"

#include "tensorflow/core/util/ctc/ctc_beam_search.h"

#include "kenlm/lm/model.hh"

typedef lm::ngram::QuantArrayTrieModel Model;

struct KenLMBeamState {
  float language_model_score;
  float score;
  float delta_score;
  int num_words;
  std::string incomplete_word;
  TrieNode *incomplete_word_trie_node;
  Model::State model_state;
};

class KenLMBeamScorer : public tensorflow::ctc::BaseBeamScorer<KenLMBeamState> {
 public:
  KenLMBeamScorer(const std::string &kenlm_path, const std::string &trie_path,
                  const std::string &alphabet_path, float lm_weight,
                  float valid_word_count_weight)
    : model_(kenlm_path.c_str(), GetLMConfig())
    , alphabet_(alphabet_path.c_str())
    , lm_weight_(lm_weight)
    , valid_word_count_weight_(valid_word_count_weight)
  {
    std::ifstream in(trie_path, std::ios::in);
    TrieNode::ReadFromStream(in, trieRoot_, alphabet_.GetSize());

    // low probability for OOV words
    oov_score_ = -10.0;
  }

  virtual ~KenLMBeamScorer() {
    delete trieRoot_;
  }

  // State initialization.
  void InitializeState(KenLMBeamState* root) const {
    root->language_model_score = 0.0f;
    root->score = 0.0f;
    root->delta_score = 0.0f;
    root->incomplete_word.clear();
    root->num_words = 0;
    root->incomplete_word_trie_node = trieRoot_;
    root->model_state = model_.BeginSentenceState();
  }
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  void ExpandState(const KenLMBeamState& from_state, int /*from_label*/,
                         KenLMBeamState* to_state, int to_label) const {
    CopyState(from_state, to_state);

    if (!alphabet_.IsSpace(to_label)) {
      to_state->incomplete_word += alphabet_.StringFromLabel(to_label);
      TrieNode *trie_node = from_state.incomplete_word_trie_node;

      // If we have no valid prefix we assume a very low log probability
      float min_unigram_score = oov_score_;
      // If prefix does exist
      if (trie_node != nullptr) {
        trie_node = trie_node->GetChildAt(to_label);
        to_state->incomplete_word_trie_node = trie_node;

        if (trie_node != nullptr) {
          min_unigram_score = trie_node->GetMinUnigramScore();
        }
      }
      // TODO try two options
      // 1) unigram score added up to language model scare
      // 2) langugage model score of (preceding_words + unigram_word)
      to_state->score = min_unigram_score + to_state->language_model_score;
      to_state->delta_score = to_state->score - from_state.score;
    } else {
      auto word_index = WordIndex(to_state->incomplete_word);
      float lm_score_delta = ScoreIncompleteWord(from_state.model_state,
                                                 word_index,
                                                 to_state->model_state);
      // Give fixed word bonus
      if (!IsOOV(word_index)) {
        to_state->language_model_score += valid_word_count_weight_;
      }
      to_state->num_words += 1;
      UpdateWithLMScore(to_state, lm_score_delta);
      ResetIncompleteWord(to_state);
    }
  }
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  void ExpandStateEnd(KenLMBeamState* state) const {
    float lm_score_delta = 0.0f;
    Model::State out;
    if (state->incomplete_word.size() > 0) {
      lm_score_delta += ScoreIncompleteWord(state->model_state,
                                            WordIndex(state->incomplete_word),
                                            out);
      ResetIncompleteWord(state);
      state->model_state = out;
    }
    lm_score_delta += model_.FullScore(state->model_state,
                                       model_.GetVocabulary().EndSentence(),
                                       out).prob;
    UpdateWithLMScore(state, lm_score_delta);


    // This is a bit of a hack. In order to implement length normalization, we
    // compute the final state score here (and not in GetStateEndExpansionScore)
    // and then set the state delta score to the value that would normalize
    // the state score when added to it. This way, we can normalize the internal
    // scores in TensorFlow's CTC code when it adds the final state expansion
    // score to this beam's score.
    state->score += lm_weight_ * state->delta_score;
    if (state->num_words > 0) {
      float normalized_score = state->score / (float)state->num_words;
      state->delta_score = normalized_score - state->score;
    }
  }
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  float GetStateExpansionScore(const KenLMBeamState& state,
                               float previous_score) const {
    return lm_weight_ * state.delta_score + previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  float GetStateEndExpansionScore(const KenLMBeamState& state) const {
    return state.delta_score;
  }

  void SetLMWeight(float lm_weight) {
    this->lm_weight_ = lm_weight;
  }

  void SetValidWordCountWeight(float valid_word_count_weight) {
    this->valid_word_count_weight_ = valid_word_count_weight;
  }

 private:
  Model model_;
  Alphabet alphabet_;
  TrieNode *trieRoot_;
  float lm_weight_;
  float valid_word_count_weight_;
  float oov_score_;

  lm::ngram::Config GetLMConfig() const {
    lm::ngram::Config config;
    config.load_method = util::POPULATE_OR_READ;
    return config;
  }

  void UpdateWithLMScore(KenLMBeamState *state, float lm_score_delta) const {
    float previous_score = state->score;
    state->language_model_score += lm_score_delta;
    state->score = state->language_model_score;
    state->delta_score = state->language_model_score - previous_score;
  }

  void ResetIncompleteWord(KenLMBeamState *state) const {
    state->incomplete_word.clear();
    state->incomplete_word_trie_node = trieRoot_;
  }

  lm::WordIndex WordIndex(const std::string& word) const {
    return model_.GetVocabulary().Index(word);
  }

  bool IsOOV(const lm::WordIndex& word) const {
    auto &vocabulary = model_.GetVocabulary();
    return word == vocabulary.NotFound();
  }

  float ScoreIncompleteWord(const Model::State& model_state,
                            const lm::WordIndex& word,
                            Model::State& out) const {
    return model_.FullScore(model_state, word, out).prob;
  }

  void CopyState(const KenLMBeamState& from, KenLMBeamState* to) const {
    to->language_model_score = from.language_model_score;
    to->score = from.score;
    to->delta_score = from.delta_score;
    to->incomplete_word = from.incomplete_word;
    to->incomplete_word_trie_node = from.incomplete_word_trie_node;
    to->model_state = from.model_state;
  }
};

#endif /* BEAM_SEARCH_H */