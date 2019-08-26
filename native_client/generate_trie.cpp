#include <algorithm>
#include <iostream>
#include <string>

#include "ctcdecode/scorer.h"
#include "alphabet.h"

using namespace std;

int generate_trie(const char* alphabet_path, const char* kenlm_path, const char* trie_path) {
  Alphabet alphabet;
  int err = alphabet.init(alphabet_path);
  if (err != 0) {
    return err;
  }
  Scorer scorer;
  err = scorer.init(0.0, 0.0, kenlm_path, "", alphabet);
  if (err != 0) {
    return err;
  }
  scorer.save_dictionary(trie_path);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <alphabet> <lm_model> <trie_path>" << std::endl;
    return -1;
  }

  return generate_trie(argv[1], argv[2], argv[3]);
}
