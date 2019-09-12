#ifdef _MSC_VER
  #include <stdlib.h>
  #include <io.h>
  #include <windows.h> 

  #define R_OK    4       /* Read permission.  */
  #define W_OK    2       /* Write permission.  */ 
  #define F_OK    0       /* Existence.  */

  #define access _access

#else          /* _MSC_VER  */
  #include <unistd.h>
#endif

#include "scorer.h"
#include <iostream>
#include <fstream>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"

#include "decoder_utils.h"

using namespace lm::ngram;

static const int32_t MAGIC = 'TRIE';
static const int32_t FILE_VERSION = 4;

int
Scorer::init(double alpha,
             double beta,
             const std::string& lm_path,
             const std::string& trie_path,
             const Alphabet& alphabet)
{
  reset_params(alpha, beta);
  alphabet_ = alphabet;
  setup(lm_path, trie_path);
  return 0;
}

int
Scorer::init(double alpha,
             double beta,
             const std::string& lm_path,
             const std::string& trie_path,
             const std::string& alphabet_config_path)
{
  reset_params(alpha, beta);
  int err = alphabet_.init(alphabet_config_path.c_str());
  if (err != 0) {
    return err;
  }
  setup(lm_path, trie_path);
  return 0;
}

void Scorer::setup(const std::string& lm_path, const std::string& trie_path)
{
  // (Re-)Initialize character map
  char_map_.clear();

  SPACE_ID_ = alphabet_.GetSpaceLabel();

  for (int i = 0; i < alphabet_.GetSize(); i++) {
    // The initial state of FST is state 0, hence the index of chars in
    // the FST should start from 1 to avoid the conflict with the initial
    // state, otherwise wrong decoding results would be given.
    char_map_[alphabet_.StringFromLabel(i)] = i + 1;
  }

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

    if (!is_character_based_) {
      fst::FstReadOptions opt;
      opt.mode = fst::FstReadOptions::MAP;
      opt.source = trie_path;
      dictionary.reset(FstType::Read(fin, opt));
    }
  }

  max_order_ = language_model_->Order();
}

void Scorer::save_dictionary(const std::string& path)
{
  std::ofstream fout(path, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
  fout.write(reinterpret_cast<const char*>(&FILE_VERSION), sizeof(FILE_VERSION));
  fout.write(reinterpret_cast<const char*>(&is_character_based_), sizeof(is_character_based_));
  if (!is_character_based_) {
    fst::FstWriteOptions opt;
    opt.align = true;
    opt.source = path;
    dictionary->Write(fout, opt);
  }
}

double Scorer::get_log_cond_prob(const std::vector<std::string>& words,
                                 bool bos,
                                 bool eos)
{
  return get_log_cond_prob(words.begin(), words.end(), bos, eos);
}

double Scorer::get_log_cond_prob(const std::vector<std::string>::const_iterator& begin,
                                 const std::vector<std::string>::const_iterator& end,
                                 bool bos,
                                 bool eos)
{
  const auto& vocab = language_model_->BaseVocabulary();
  lm::ngram::State state_vec[2];
  lm::ngram::State *in_state = &state_vec[0];
  lm::ngram::State *out_state = &state_vec[1];

  if (bos) {
    language_model_->BeginSentenceWrite(in_state);
  } else {
    language_model_->NullContextWrite(in_state);
  }

  double cond_prob = 0.0;
  for (auto it = begin; it != end; ++it) {
    lm::WordIndex word_index = vocab.Index(*it);

    // encounter OOV
    if (word_index == lm::kUNK) {
      return OOV_SCORE;
    }

    cond_prob = language_model_->BaseScore(in_state, word_index, out_state);
    std::swap(in_state, out_state);
  }

  if (eos) {
    cond_prob = language_model_->BaseScore(in_state, vocab.EndSentence(), out_state);
  }

  // return loge prob
  return cond_prob/NUM_FLT_LOGE;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words)
{
  // For a given sentence (`words`), return sum of LM scores over windows on
  // sentence. For example, given the sentence:
  //
  //    there once was an ugly barnacle
  //
  // And a language model with max_order_ = 3, this function will return the sum
  // of the following scores:
  //
  //    there                  | <s>
  //    there   once           | <s>
  //    there   once     was
  //    once    was      an
  //    was     an       ugly
  //    an      ugly     barnacle
  //    ugly    barnacle </s>
  //
  // This is used in the decoding process to compute the LM contribution for a
  // given beam's accumulated score, so that it can be removed and only the
  // acoustic model contribution can be returned as a confidence score for the
  // transcription. See DecoderState::decode.
  const int sent_len = words.size();

  double score = 0.0;
  for (int win_start = 0, win_end = 1; win_end <= sent_len+1; ++win_end) {
    const int win_size = win_end - win_start;
    bool bos = win_size < max_order_;
    bool eos = win_end == sent_len + 1;

    // The last window goes one past the end of the words vector as passing the
    // EOS=true flag counts towards the length of the scored sentence, so we
    // adjust the win_end index here to not go over bounds.
    score += get_log_cond_prob(words.begin() + win_start,
                               words.begin() + (eos ? win_end - 1 : win_end),
                               bos,
                               eos);

    // Only increment window start position after we have a full window
    if (win_size == max_order_) {
      win_start++;
    }
  }

  return score / NUM_FLT_LOGE;
}

void Scorer::reset_params(float alpha, float beta)
{
  this->alpha = alpha;
  this->beta = beta;
}

std::vector<std::string> Scorer::split_labels(const std::vector<int>& labels)
{
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

std::vector<std::string> Scorer::make_ngram(PathTrie* prefix)
{
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
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void Scorer::fill_dictionary(const std::vector<std::string>& vocabulary, bool add_space)
{
  // ConstFst is immutable, so we need to use a MutableFst to create the trie,
  // and then we convert to a ConstFst for the decoder and for storing on disk.
  fst::StdVectorFst dictionary;
  // For each unigram convert to ints and put in trie
  for (const auto& word : vocabulary) {
    add_word_to_dictionary(word, char_map_, add_space, SPACE_ID_ + 1, &dictionary);
  }

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST deterministic, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  std::unique_ptr<fst::StdVectorFst> new_dict(new fst::StdVectorFst);

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict.get());

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict.get());

  // Now we convert the MutableFst to a ConstFst (Scorer::FstType) via its ctor
  std::unique_ptr<FstType> converted(new FstType(*new_dict));
  this->dictionary = std::move(converted);
}
