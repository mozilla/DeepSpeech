#include <iostream>
#include <memory>
#include <string>

#include "alphabet.h"
#include "trie_node.h"

int main(int argc, char** argv)
{
  const char* trie_path     = argv[1];
  const char* alphabet_path = argv[2];

  printf("Loading trie(%s) and alphabet(%s)\n", trie_path, alphabet_path);

  Alphabet alphabet_ = Alphabet(alphabet_path);
  TrieNode *trieRoot_;

  std::ifstream in(trie_path, std::ios::in | std::ios::binary);
  TrieNode::ReadFromStream(in, trieRoot_, alphabet_.GetSize());

  return 0;
}
