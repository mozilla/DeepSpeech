#include "scorer.h"

#include <unistd.h>
#include <iostream>
#include <fstream>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"

#include "decoder_utils.h"

using namespace lm::ngram;

static const int MAGIC = 'TRIE';
static const int FILE_VERSION = 3;

Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::string& trie_path,
               const Alphabet& alphabet)
  : dictionary()
  , language_model_()
  , is_character_based_(true)
  , max_order_(0)
  , alphabet_(alphabet)
{
  reset_params(alpha, beta);

  char_map_.clear();

  SPACE_ID_ = alphabet_.GetSpaceLabel();

  for (int i = 0; i < alphabet_.GetSize(); i++) {
    // The initial state of FST is state 0, hence the index of chars in
    // the FST should start from 1 to avoid the conflict with the initial
    // state, otherwise wrong decoding results would be given.
    char_map_[alphabet_.StringFromLabel(i)] = i + 1;
  }

  setup(lm_path, trie_path);
}

Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::string& trie_path,
               const char* alphabet_config_path)
  : Scorer(alpha, beta, lm_path, trie_path, Alphabet(alphabet_config_path))
{

}

Scorer::~Scorer() {
}

void Scorer::setup(const std::string& lm_path, const std::string& trie_path) {
  // load language model
  const char* filename = lm_path.c_str();
  VALID_CHECK_EQ(access(filename, R_OK), 0, "Invalid language model path");

  bool has_trie = trie_path.size() && access(trie_path.c_str(), R_OK) == 0;

  lm::ngram::Config config;

  if (!has_trie) { // no trie was specified, build it now
    RetrieveStrEnumerateVocab enumerate;
    config.enumerate_vocab = &enumerate;
    language_model_.reset(lm::ngram::LoadVirtual(filename, config));
    auto vocab = enumerate.vocabulary;
    for (size_t i = 0; i < vocab.size(); ++i) {
      if (is_character_based_ && vocab[i] != UNK_TOKEN &&
          vocab[i] != START_TOKEN && vocab[i] != END_TOKEN &&
          get_utf8_str_len(enumerate.vocabulary[i]) > 1) {
        is_character_based_ = false;
      }
    }
    // fill the dictionary for FST
    if (!is_character_based()) {
      fill_dictionary(vocab, true);
    }
  } else {
    language_model_.reset(lm::ngram::LoadVirtual(filename, config));

    // Read metadata and trie from file
    std::ifstream fin(trie_path, std::ios::binary);

    int magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MAGIC) {
      std::cerr << "Error: Can't parse trie file, invalid header. Try updating "
                   "your trie file." << std::endl;
      throw 1;
    }

    int version;
    fin.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != FILE_VERSION) {
      std::cerr << "Error: Trie file version mismatch (" << version
                << " instead of expected " << FILE_VERSION
                << "). Update your trie file."
                << std::endl;
      throw 1;
    }

    fin.read(reinterpret_cast<char*>(&is_character_based_), sizeof(is_character_based_));

    fst::FstReadOptions opt;
    dictionary.reset(fst::StdVectorFst::Read(fin, opt));
  }

  max_order_ = language_model_->Order();
}

void Scorer::save_dictionary(const std::string& path) {
  std::ofstream fout(path, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
  fout.write(reinterpret_cast<const char*>(&FILE_VERSION), sizeof(FILE_VERSION));
  fout.write(reinterpret_cast<const char*>(&is_character_based_), sizeof(is_character_based_));
  fst::FstWriteOptions opt;
  dictionary->Write(fout, opt);
}

double Scorer::get_log_cond_prob(const std::vector<std::string>& words) {
  double cond_prob = OOV_SCORE;
  lm::ngram::State state, tmp_state, out_state;
  // avoid to inserting <s> in begin
  language_model_->NullContextWrite(&state);
  for (size_t i = 0; i < words.size(); ++i) {
    lm::WordIndex word_index = language_model_->BaseVocabulary().Index(words[i]);
    // encounter OOV
    if (word_index == 0) {
      return OOV_SCORE;
    }
    cond_prob = language_model_->BaseScore(&state, word_index, &out_state);
    tmp_state = state;
    state = out_state;
    out_state = tmp_state;
  }
  // return  loge prob
  return cond_prob/NUM_FLT_LOGE;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words) {
  std::vector<std::string> sentence;
  if (words.size() == 0) {
    for (size_t i = 0; i < max_order_; ++i) {
      sentence.push_back(START_TOKEN);
    }
  } else {
    for (size_t i = 0; i < max_order_ - 1; ++i) {
      sentence.push_back(START_TOKEN);
    }
    sentence.insert(sentence.end(), words.begin(), words.end());
  }
  sentence.push_back(END_TOKEN);
  return get_log_prob(sentence);
}

double Scorer::get_log_prob(const std::vector<std::string>& words) {
  assert(words.size() > max_order_);
  double score = 0.0;
  for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
    std::vector<std::string> ngram(words.begin() + i,
                                   words.begin() + i + max_order_);
    score += get_log_cond_prob(ngram);
  }
  return score;
}

void Scorer::reset_params(float alpha, float beta) {
  this->alpha = alpha;
  this->beta = beta;
}

std::vector<std::string> Scorer::split_labels(const std::vector<int>& labels) {
  if (labels.empty()) return {};

  std::string s = alphabet_.LabelsToString(labels);
  std::vector<std::string> words;
  if (is_character_based_) {
    words = split_utf8_str(s);
  } else {
    words = split_str(s, " ");
  }
  return words;
}

std::vector<std::string> Scorer::make_ngram(PathTrie* prefix) {
  std::vector<std::string> ngram;
  PathTrie* current_node = prefix;
  PathTrie* new_node = nullptr;

  for (int order = 0; order < max_order_; order++) {
    std::vector<int> prefix_vec;
    std::vector<int> prefix_steps;

    if (is_character_based_) {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, SPACE_ID_, 1);
      current_node = new_node;
    } else {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, SPACE_ID_);
      current_node = new_node->parent;  // Skipping spaces
    }

    // reconstruct word
    std::string word = alphabet_.LabelsToString(prefix_vec);
    ngram.push_back(word);

    if (new_node->character == -1) {
      // No more spaces, but still need order
      for (int i = 0; i < max_order_ - order - 1; i++) {
        ngram.push_back(START_TOKEN);
      }
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void Scorer::fill_dictionary(const std::vector<std::string>& vocabulary, bool add_space) {
  fst::StdVectorFst dictionary;
  // For each unigram convert to ints and put in trie
  for (const auto& word : vocabulary) {
    add_word_to_dictionary(word, char_map_, add_space, SPACE_ID_ + 1, &dictionary);
  }

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);
  this->dictionary.reset(new_dict);
}
