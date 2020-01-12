#include <algorithm>
#include <iostream>
#include <string>

#include "ctcdecode/scorer.h"
#include "alphabet.h"

using namespace std;

int generate_trie(const char* alphabet_path, const char* kenlm_path, const char* trie_path) {
  string word;
  vector<string> vocabulary;
  ifstream fin(kenlm_path);
  while (!fin.eof()) {
    fin >> word;
    vocabulary.push_back(word);
  }
  Alphabet alphabet;
  alphabet.init(alphabet_path);
  Scorer scorer;
  scorer.init(1.0, 1.0, "", "", alphabet);
  scorer.is_utf8_mode_ = false;
  scorer.fill_dictionary(vocabulary);
  scorer.save_dictionary(trie_path);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <alphabet> <vocab_file_word_per_line> <trie_path>" << std::endl;
    return -1;
  }

  return generate_trie(argv[1], argv[2], argv[3]);
}
