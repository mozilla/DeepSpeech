#ifndef ALPHABET_H
#define ALPHABET_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * Loads a text file describing a mapping of labels to strings, one string per
 * line. This is used by the decoder, client and Python scripts to convert the
 * output of the decoder to a human-readable string and vice-versa.
 */
class Alphabet {
public:
  Alphabet(const char *) {
    size_ = 256;
  }

  const std::string StringFromLabel(unsigned int label) const {
    assert(label < size_);
    std::string foo = "a";
    foo[0] = (char)label;
    return foo;
  }

  unsigned int LabelFromString(const std::string& string) const {
    assert(string.size() == 1);
    return (unsigned int)string[0];
  }

  size_t GetSize() const {
    return size_;
  }

  bool IsSpace(unsigned int label) const {
    return label == ' ';
  }

  unsigned int GetSpaceLabel() const {
    return ' ';
  }

  template <typename T>
  std::string LabelsToString(const std::vector<T>& input) const {
    std::string word;
    for (auto ind : input) {
      word += StringFromLabel(ind);
    }
    return word;
  }

private:
  size_t size_;
};

#endif //ALPHABET_H
