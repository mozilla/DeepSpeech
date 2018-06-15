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
#include <iostream>
#include <istream>
#include <limits>
#include <locale>
#include <string>

#include <boost/locale/encoding_utf.hpp>

class TrieNode {
public:
  static const int MAGIC = 'TRIE';
  static const int FILE_VERSION = 1;

  TrieNode(int vocab_size)
    : vocab_size_(vocab_size)
    , prefixCount_(0)
    , min_score_word_(0)
    , min_unigram_score_(std::numeric_limits<float>::max())
    {
      children_ = new TrieNode*[vocab_size_]();
    }

  ~TrieNode() {
    for (int i = 0; i < vocab_size_; i++) {
      delete children_[i];
    }
    delete children_;
  }

  void WriteToStream(std::ostream& os) const {
    os << MAGIC << std::endl << FILE_VERSION << std::endl << vocab_size_ << std::endl;
    WriteNodeAndChildren(os);
  }

  static void ReadFromStream(std::istream& is, TrieNode* &obj, int vocab_size) {
    int magic;
    is >> magic;
    if (magic != MAGIC) {
      std::cerr << "Error: Can't parse trie file, invalid header. Try updating "
                   "your trie file." << std::endl;
      obj = nullptr;
      return;
    }

    int version;
    is >> version;
    if (version != FILE_VERSION) {
      std::cerr << "Error: Trie file version mismatch. Update your trie file."
                << std::endl;
      obj = nullptr;
      return;
    }

    int fileVocabSize;
    is >> fileVocabSize;
    if (fileVocabSize != vocab_size) {
      std::cerr << "Error: Mismatching alphabet size in trie file and alphabet "
                   "file. Trie file will not be loaded." << std::endl;
      obj = nullptr;
      return;
    }

    ReadPrefixAndNode(is, obj, vocab_size);
  }

  void Insert(const std::string& word, std::function<int (const std::string&)> translator,
              lm::WordIndex lm_word, float unigram_score) {
    // All strings are UTF-8 encoded at the API boundaries. We need to iterate
    // on codepoints in order to support multi-byte characters, so we convert
    // to UCS-4 to extract the first codepoint, then the codepoint back to
    // UTF-8 to translate it into a vocabulary index.

    //TODO We should normalize the input first, and possibly iterate by grapheme
    //     instead of codepoint for languages that don't have composed versions
    //     of multi-codepoint characters. This requires extra dependencies so
    //     leaving as a future improvement when the need arises.
    using namespace boost::locale::conv;
    std::u32string codepoints = utf_to_utf<char32_t, char>(word, method_type::stop);
    Insert(codepoints.begin(), translator, lm_word, unigram_score);
  }

  int GetPrefixCount() const {
    return prefixCount_;
  }

  lm::WordIndex GetMinScoreWordIndex() const {
    return min_score_word_;
  }

  float GetMinUnigramScore() const {
    return min_unigram_score_;
  }

  TrieNode *GetChildAt(int vocabIndex) {
    return children_[vocabIndex];
  }

private:
  int vocab_size_;
  int prefixCount_;
  lm::WordIndex min_score_word_;
  float min_unigram_score_;
  TrieNode **children_;

  void Insert(const std::u32string::iterator& codepoints,
              std::function<int (const std::string&)> translator,
              lm::WordIndex lm_word, float unigram_score) {
    using namespace boost::locale::conv;
    char32_t firstCodepoint[2] = {*codepoints, 0};
    std::string firstCodepoint_utf8 =
      utf_to_utf<char, char32_t>(firstCodepoint, method_type::stop);

    prefixCount_++;
    if (unigram_score < min_unigram_score_) {
      min_unigram_score_ = unigram_score;
      min_score_word_ = lm_word;
    }
    if (firstCodepoint[0] != 0) {
      int vocabIndex = translator(firstCodepoint_utf8);
      if (children_[vocabIndex] == nullptr) {
        children_[vocabIndex] = new TrieNode(vocab_size_);
      }
      children_[vocabIndex]->Insert(codepoints+1, translator, lm_word, unigram_score);
    }
  }

  void WriteNode(std::ostream& os) const {
    os << prefixCount_ << std::endl;
    os << min_score_word_ << std::endl;
    os << min_unigram_score_ << std::endl;
  }

  void ReadNode(std::istream& is, int first_input) {
    prefixCount_ = first_input;
    is >> min_score_word_;
    is >> min_unigram_score_;
  }

  void WriteNodeAndChildren(std::ostream& os) const {
    WriteNode(os);
    for (int i = 0; i < vocab_size_; i++) {
      if (children_[i] == nullptr) {
        os << -1 << std::endl;
      } else {
        // Recursive call
        children_[i]->WriteNodeAndChildren(os);
      }
    }
  }

  static void ReadPrefixAndNode(std::istream& is, TrieNode* &obj, int vocab_size) {
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
      ReadPrefixAndNode(is, obj->children_[i], vocab_size);
    }
  }
};

#endif //TRIE_NODE_H
