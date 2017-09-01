#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

#include "lm/model.hh"
#include "trie_node.h"
#include "alphabet.h"

typedef lm::ngram::ProbingModel Model;

lm::WordIndex GetWordIndex(const Model& model, const std::string& word) {
  lm::WordIndex vocab;
  vocab = model.GetVocabulary().Index(word);
  return vocab;
}

float ScoreWord(const Model& model, lm::WordIndex vocab) {
  Model::State in_state = model.NullContextState();
  Model::State out;
  lm::FullScoreReturn full_score_return;
  full_score_return = model.FullScore(in_state, vocab, out);
  return full_score_return.prob;
}

int generate_trie(const char* alphabet_path, const char* kenlm_path, const char* vocab_path, const char* trie_path) {
  Alphabet a(alphabet_path);

  lm::ngram::Config config;
  config.load_method = util::POPULATE_OR_READ;
  Model model(kenlm_path, config);
  TrieNode root(a.GetSize());

  std::ifstream ifs;
  ifs.open(vocab_path, std::ifstream::in);

  if (!ifs.is_open()) {
    std::cout << "unable to open vocabulary" << std::endl;
    return -1;
  }

  std::ofstream ofs;
  ofs.open(trie_path);

  std::string word;
  while (ifs >> word) {
    for_each(word.begin(), word.end(), [](char& a) { a = tolower(a); });
    lm::WordIndex vocab = GetWordIndex(model, word);
    float unigram_score = ScoreWord(model, vocab);
    root.Insert(word.c_str(), [&a](char c) {
                  return a.LabelFromString(string(1, c));
                }, vocab, unigram_score);
  }

  root.WriteToStream(ofs);
  ifs.close();
  ofs.close();
  return 0;
}

int main(void) {
  return generate_trie("/Users/remorais/Development/DeepSpeech/data/alphabet.txt",
                       "/Users/remorais/Development/DeepSpeech/data/lm/lm.binary",
                       "/Users/remorais/Development/DeepSpeech/data/lm/vocab.txt",
                       "/Users/remorais/Development/DeepSpeech/data/lm/trie");
}
