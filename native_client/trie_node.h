/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TRIE_NODE_H
#define TRIE_NODE_H

#include "lm/model.hh"

#include <functional>
#include <istream>
#include <iostream>
#include <limits>

class TrieNode {
public:
  TrieNode(int vocab_size)
    : vocab_size(vocab_size)
    , prefixCount(0)
    , min_score_word(0)
    , min_unigram_score(std::numeric_limits<float>::max())
    {
      children = new TrieNode*[vocab_size]();
    }

  ~TrieNode() {
    for (int i = 0; i < vocab_size; i++) {
      delete children[i];
    }
    delete children;
  }

  void WriteToStream(std::ostream& os) {
    WriteNode(os);
    for (int i = 0; i < vocab_size; i++) {
      if (children[i] == nullptr) {
        os << -1 << std::endl;
      } else {
        // Recursive call
        children[i]->WriteToStream(os);
      }
    }
  }

  static void ReadFromStream(std::istream& is, TrieNode* &obj, int vocab_size) {
    int prefixCount;
    is >> prefixCount;

    if (prefixCount == -1) {
      // This is an undefined child
      obj = nullptr;
      return;
    }

    obj = new TrieNode(vocab_size);
    obj->ReadNode(is, prefixCount);
    for (int i = 0; i < vocab_size; i++) {
      // Recursive call
      ReadFromStream(is, obj->children[i], vocab_size);
    }
  }

  void Insert(const char* word, std::function<int (char)> translator,
              lm::WordIndex lm_word, float unigram_score) {
    char wordCharacter = *word;
    prefixCount++;
    if (unigram_score < min_unigram_score) {
      min_unigram_score = unigram_score;
      min_score_word = lm_word;
    }
    if (wordCharacter != '\0') {
      int vocabIndex = translator(wordCharacter);
      TrieNode *child = children[vocabIndex];
      if (child == nullptr)
        child = children[vocabIndex] = new TrieNode(vocab_size);
      child->Insert(word + 1, translator, lm_word, unigram_score);
    }
  }

  int GetFrequency() {
    return prefixCount;
  }

  lm::WordIndex GetMinScoreWordIndex() {
    return min_score_word;
  }

  float GetMinUnigramScore() {
    return min_unigram_score;
  }

  TrieNode *GetChildAt(int vocabIndex) {
    return children[vocabIndex];
  }

private:
  int vocab_size;
  int prefixCount;
  lm::WordIndex min_score_word;
  float min_unigram_score;
  TrieNode **children;

  void WriteNode(std::ostream& os) const {
    os << prefixCount << std::endl;
    os << min_score_word << std::endl;
    os << min_unigram_score << std::endl;
  }

  void ReadNode(std::istream& is, int first_input) {
    prefixCount = first_input;
    is >> min_score_word;
    is >> min_unigram_score;
  }

};

#endif //TRIE_NODE_H
