#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

#include "lm/model.hh"
#include "trie_node.h"
#include "alphabet.h"

typedef lm::ngram::ProbingModel Model;

lm::WordIndex GetWordIndex(const Model& model, const std::string& word) {
  return model.GetVocabulary().Index(word);
}

float ScoreWord(const Model& model, lm::WordIndex word_index) {
  // We don't need to keep state here as we're scoring the words individually.
  Model::State out;
  return model.FullScore(model.NullContextState(), word_index, out).prob;
}

int generate_trie(const char* alphabet_path, const char* kenlm_path, const char* vocab_path, const char* trie_path) {
  Alphabet a(alphabet_path);

  lm::ngram::Config config;
  config.load_method = util::POPULATE_OR_READ;
  Model model(kenlm_path, config);
  TrieNode root(a.GetSize());

  std::ifstream ifs(vocab_path, std::ifstream::in);
  if (!ifs) {
    std::cerr << "unable to open vocabulary file " << vocab_path << std::endl;
    return -1;
  }

  std::ofstream ofs(trie_path);
  if (!ofs) {
    std::cerr << "unable to open output file " << trie_path << std::endl;
    return -1;
  }

  std::string word;
  while (ifs >> word) {
    lm::WordIndex word_index = GetWordIndex(model, word);
    float unigram_score = ScoreWord(model, word_index);
    root.Insert(word,
                [&a](const std::string& c) {
                  return a.LabelFromString(c);
                },
                word_index, unigram_score);
  }

  root.WriteToStream(ofs);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <alphabet> <lm_model> <vocabulary> <trie_path>" << std::endl;
    return -1;
  }

  return generate_trie(argv[1], argv[2], argv[3], argv[4]);
}
