#ifndef SCORER_H_
#define SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

#include "path_trie.h"
#include "alphabet.h"
#include "deepspeech.h"

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 */
class Scorer {
public:
  using FstType = PathTrie::FstType;

  Scorer() = default;
  ~Scorer() = default;

  // disallow copying
  Scorer(const Scorer&) = delete;
  Scorer& operator=(const Scorer&) = delete;

  int init(const std::string &lm_path,
           const Alphabet &alphabet);

  int init(const std::string &lm_path,
           const std::string &alphabet_config_path);

  double get_log_cond_prob(const std::vector<std::string> &words,
                           bool bos = false,
                           bool eos = false);

  double get_log_cond_prob(const std::vector<std::string>::const_iterator &begin,
                           const std::vector<std::string>::const_iterator &end,
                           bool bos = false,
                           bool eos = false);

  // return the max order
  size_t get_max_order() const { return max_order_; }

  // return true if the language model is character based
  bool is_utf8_mode() const { return is_utf8_mode_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta);

  // force set UTF-8 mode, ignore value read from file
  void set_utf8_mode(bool utf8) { is_utf8_mode_ = utf8; }

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix);

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels_into_scored_units(const std::vector<unsigned int> &labels);

  void set_alphabet(const Alphabet& alphabet);

  // save dictionary in file
  bool save_dictionary(const std::string &path, bool append_instead_of_overwrite=false);

  // return weather this step represents a boundary where beam scoring should happen
  bool is_scoring_boundary(PathTrie* prefix, size_t new_label);

  // fill dictionary FST from a vocabulary
  void fill_dictionary(const std::unordered_set<std::string> &vocabulary);

  // load language model from given path
  int load_lm(const std::string &lm_path);

  // language model weight
  double alpha = 0.;
  // word insertion weight
  double beta = 0.;

  // pointer to the dictionary of FST
  std::unique_ptr<FstType> dictionary;

protected:
  // necessary setup after setting alphabet
  void setup_char_map();

  int load_trie(std::ifstream& fin, const std::string& file_path);

private:
  std::unique_ptr<lm::base::Model> language_model_;
  bool is_utf8_mode_ = true;
  size_t max_order_ = 0;

  int SPACE_ID_;
  Alphabet alphabet_;
  std::unordered_map<std::string, int> char_map_;
};

#endif  // SCORER_H_
