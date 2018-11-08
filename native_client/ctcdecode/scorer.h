#ifndef SCORER_H_
#define SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

#include "path_trie.h"
#include "alphabet.h"

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

// Implement a callback to retrieve the dictionary of language model.
class RetrieveStrEnumerateVocab : public lm::EnumerateVocab {
public:
  RetrieveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */
class Scorer {
public:
  Scorer(double alpha,
         double beta,
         const std::string &lm_path,
         const std::string &trie_path,
         const Alphabet &alphabet);
  Scorer(double alpha,
         double beta,
         const std::string &lm_path,
         const std::string &trie_path,
         const char* alphabet_config_path);
  ~Scorer();

  double get_log_cond_prob(const std::vector<std::string> &words);

  double get_sent_log_prob(const std::vector<std::string> &words);

  // return the max order
  size_t get_max_order() const { return max_order_; }

  // retrun true if the language model is character based
  bool is_character_based() const { return is_character_based_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta);

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix);

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels(const std::vector<int> &labels);

  // save dictionary in file
  void save_dictionary(const std::string &path);

  // language model weight
  double alpha;
  // word insertion weight
  double beta;

  // pointer to the dictionary of FST
  std::unique_ptr<fst::StdVectorFst> dictionary;

protected:
  // necessary setup: load language model, fill FST's dictionary
  void setup(const std::string &lm_path, const std::string &trie_path);

  // load language model from given path
  void load_lm(const std::string &lm_path);

  // fill dictionary for FST
  void fill_dictionary(const std::vector<std::string> &vocabulary, bool add_space);

  double get_log_prob(const std::vector<std::string> &words);

private:
  std::unique_ptr<lm::base::Model> language_model_;
  bool is_character_based_;
  size_t max_order_;

  int SPACE_ID_;
  Alphabet alphabet_;
  std::unordered_map<std::string, int> char_map_;
};

#endif  // SCORER_H_
