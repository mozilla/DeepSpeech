#ifndef ALPHABET_H
#define ALPHABET_H

#include <cassert>
#include <fstream>
#include <string>
#include <unordered_map>

/*
 * Loads a text file describing a mapping of labels to strings, one string per
 * line. This is used by the decoder, client and Python scripts to convert the
 * output of the decoder to a human-readable string and vice-versa.
 */
class Alphabet {
public:
  Alphabet(const char *config_file) {
    std::ifstream in(config_file, std::ios::in);
    int label = 0;
    for (std::string line; std::getline(in, line); ++label) {
      label_to_str_[label] = line;
      str_to_label_[line] = label;
    }
    size_ = label;
    in.close();
  }

  std::string StringFromLabel(int label) {
    assert(label < size_);
    return label_to_str_[label];
  }

  int LabelFromString(std::string string) {
    return str_to_label_[string];
  }

  int GetSize() {
    return size_;
  }

private:
  int size_;
  std::unordered_map<int, std::string> label_to_str_;
  std::unordered_map<std::string, int> str_to_label_;
};

#endif //ALPHABET_H
