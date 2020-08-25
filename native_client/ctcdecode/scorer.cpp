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

static const int32_t MAGIC = 'TRIE';
static const int32_t FILE_VERSION = 6;

int
Scorer::init(const std::string& lm_path,
             const Alphabet& alphabet)
{
  set_alphabet(alphabet);
  return load_lm(lm_path);
}

int
Scorer::init(const std::string& lm_path,
             const std::string& alphabet_config_path)
{
  int err = alphabet_.init(alphabet_config_path.c_str());
  if (err != 0) {
    return err;
  }
  setup_char_map();
  return load_lm(lm_path);
}

void
Scorer::set_alphabet(const Alphabet& alphabet)
{
  alphabet_ = alphabet;
  setup_char_map();
}

void Scorer::setup_char_map()
{
  // (Re-)Initialize character map
  char_map_.clear();

  SPACE_ID_ = alphabet_.GetSpaceLabel();

  for (int i = 0; i < alphabet_.GetSize(); i++) {
    // The initial state of FST is state 0, hence the index of chars in
    // the FST should start from 1 to avoid the conflict with the initial
    // state, otherwise wrong decoding results would be given.
    char_map_[alphabet_.DecodeSingle(i)] = i + 1;
  }
}

int Scorer::load_lm(const std::string& lm_path)
{
  // Check if file is readable to avoid KenLM throwing an exception
  const char* filename = lm_path.c_str();
  if (access(filename, R_OK) != 0) {
    return DS_ERR_SCORER_UNREADABLE;
  }

  // Check if the file format is valid to avoid KenLM throwing an exception
  lm::ngram::ModelType model_type;
  if (!lm::ngram::RecognizeBinary(filename, model_type)) {
    return DS_ERR_SCORER_INVALID_LM;
  }

  // Load the LM
  lm::ngram::Config config;
  config.load_method = util::LoadMethod::LAZY;
  language_model_.reset(lm::ngram::LoadVirtual(filename, config));
  max_order_ = language_model_->Order();

  uint64_t package_size;
  {
    util::scoped_fd fd(util::OpenReadOrThrow(filename));
    package_size = util::SizeFile(fd.get());
  }
  uint64_t trie_offset = language_model_->GetEndOfSearchOffset();
  if (package_size <= trie_offset) {
    // File ends without a trie structure
    return DS_ERR_SCORER_NO_TRIE;
  }

  // Read metadata and trie from file
  std::ifstream fin(lm_path, std::ios::binary);
  fin.seekg(trie_offset);
  return load_trie(fin, lm_path);
}

int Scorer::load_trie(std::ifstream& fin, const std::string& file_path)
{
  int magic;
  fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic != MAGIC) {
    std::cerr << "Error: Can't parse scorer file, invalid header. Try updating "
                 "your scorer file." << std::endl;
    return DS_ERR_SCORER_INVALID_TRIE;
  }

  int version;
  fin.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != FILE_VERSION) {
    std::cerr << "Error: Scorer file version mismatch (" << version
              << " instead of expected " << FILE_VERSION
              << "). ";
    if (version < FILE_VERSION) {
      std::cerr << "Update your scorer file.";
    } else {
      std::cerr << "Downgrade your scorer file or update your version of DeepSpeech.";
    }
    std::cerr << std::endl;
    return DS_ERR_SCORER_VERSION_MISMATCH;
  }

  fin.read(reinterpret_cast<char*>(&is_utf8_mode_), sizeof(is_utf8_mode_));

  // Read hyperparameters from header
  double alpha, beta;
  fin.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
  fin.read(reinterpret_cast<char*>(&beta), sizeof(beta));
  reset_params(alpha, beta);

  fst::FstReadOptions opt;
  opt.mode = fst::FstReadOptions::MAP;
  opt.source = file_path;
  dictionary.reset(FstType::Read(fin, opt));
  return DS_ERR_OK;
}

bool Scorer::save_dictionary(const std::string& path, bool append_instead_of_overwrite)
{
  std::ios::openmode om;
  if (append_instead_of_overwrite) {
    om = std::ios::in|std::ios::out|std::ios::binary|std::ios::ate;
  } else {
    om = std::ios::out|std::ios::binary;
  }
  std::fstream fout(path, om);
  if (!fout ||fout.bad()) {
    std::cerr << "Error opening '" << path << "'" << std::endl;
    return false;
  }
  fout.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
  if (fout.bad()) {
    std::cerr << "Error writing MAGIC '" << path << "'" << std::endl;
    return false;
  }
  fout.write(reinterpret_cast<const char*>(&FILE_VERSION), sizeof(FILE_VERSION));
  if (fout.bad()) {
    std::cerr << "Error writing FILE_VERSION '" << path << "'" << std::endl;
    return false;
  }
  fout.write(reinterpret_cast<const char*>(&is_utf8_mode_), sizeof(is_utf8_mode_));
  if (fout.bad()) {
    std::cerr << "Error writing is_utf8_mode '" << path << "'" << std::endl;
    return false;
  }
  fout.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
  if (fout.bad()) {
    std::cerr << "Error writing alpha '" << path << "'" << std::endl;
    return false;
  }
  fout.write(reinterpret_cast<const char*>(&beta), sizeof(beta));
  if (fout.bad()) {
    std::cerr << "Error writing beta '" << path << "'" << std::endl;
    return false;
  }
  fst::FstWriteOptions opt;
  opt.align = true;
  opt.source = path;
  return dictionary->Write(fout, opt);
}

bool Scorer::is_scoring_boundary(PathTrie* prefix, size_t new_label)
{
  if (is_utf8_mode()) {
    if (prefix->character == -1) {
      return false;
    }
    unsigned char first_byte;
    int distance_to_boundary = prefix->distance_to_codepoint_boundary(&first_byte, alphabet_);
    int needed_bytes;
    if ((first_byte >> 3) == 0x1E) {
      needed_bytes = 4;
    } else if ((first_byte >> 4) == 0x0E) {
      needed_bytes = 3;
    } else if ((first_byte >> 5) == 0x06) {
      needed_bytes = 2;
    } else if ((first_byte >> 7) == 0x00) {
      needed_bytes = 1;
    } else {
      assert(false); // invalid byte sequence. should be unreachable, disallowed by vocabulary/trie
      return false;
    }
    return distance_to_boundary == needed_bytes;
  } else {
    return new_label == SPACE_ID_;
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

void Scorer::reset_params(float alpha, float beta)
{
  this->alpha = alpha;
  this->beta = beta;
}

std::vector<std::string> Scorer::split_labels_into_scored_units(const std::vector<unsigned int>& labels)
{
  if (labels.empty()) return {};

  std::string s = alphabet_.Decode(labels);
  std::vector<std::string> words;
  if (is_utf8_mode_) {
    words = split_into_codepoints(s);
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
    if (!current_node || current_node->character == -1) {
      break;
    }

    std::vector<unsigned int> prefix_vec;
    std::vector<unsigned int> prefix_steps;

    if (is_utf8_mode_) {
      new_node = current_node->get_prev_grapheme(prefix_vec, prefix_steps, alphabet_);
    } else {
      new_node = current_node->get_prev_word(prefix_vec, prefix_steps, alphabet_);
    }
    current_node = new_node->parent;

    // reconstruct word
    std::string word = alphabet_.Decode(prefix_vec);
    ngram.push_back(word);
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void Scorer::fill_dictionary(const std::unordered_set<std::string>& vocabulary)
{
  // ConstFst is immutable, so we need to use a MutableFst to create the trie,
  // and then we convert to a ConstFst for the decoder and for storing on disk.
  fst::StdVectorFst dictionary;
  // For each unigram convert to ints and put in trie
  for (const auto& word : vocabulary) {
    if (word != START_TOKEN && word != UNK_TOKEN && word != END_TOKEN) {
      add_word_to_dictionary(word, char_map_, is_utf8_mode_, SPACE_ID_ + 1, &dictionary);
    }
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
