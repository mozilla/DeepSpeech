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
  Alphabet() = default;
  Alphabet(const Alphabet&) = default;
  Alphabet& operator=(const Alphabet&) = default;

  int init(const char *config_file) {
    std::ifstream in(config_file, std::ios::in);
    if (!in) {
      return 1;
    }
    unsigned int label = 0;
    space_label_ = -2;
    for (std::string line; std::getline(in, line);) {
      if (line.size() == 2 && line[0] == '\\' && line[1] == '#') {
        line = '#';
      } else if (line[0] == '#') {
        continue;
      }
      //TODO: we should probably do something more i18n-aware here
      if (line == " ") {
        space_label_ = label;
      }
      label_to_str_[label] = line;
      str_to_label_[line] = label;
      ++label;
    }
    size_ = label;
    in.close();
    return 0;
  }

  int deserialize(const char* buffer, const int buffer_size) {
    int offset = 0;
    if (buffer_size - offset < sizeof(uint16_t)) {
      return 1;
    }
    uint16_t size = *(uint16_t*)(buffer + offset);
    offset += sizeof(uint16_t);
    size_ = size;

    for (int i = 0; i < size; ++i) {
      if (buffer_size - offset < sizeof(uint16_t)) {
        return 1;
      }
      uint16_t label = *(uint16_t*)(buffer + offset);
      offset += sizeof(uint16_t);

      if (buffer_size - offset < sizeof(uint16_t)) {
        return 1;
      }
      uint16_t val_len = *(uint16_t*)(buffer + offset);
      offset += sizeof(uint16_t);

      if (buffer_size - offset < val_len) {
        return 1;
      }
      std::string val(buffer+offset, val_len);
      offset += val_len;

      label_to_str_[label] = val;
      str_to_label_[val] = label;

      if (val == " ") {
        space_label_ = label;
      }
    }

    return 0;
  }

  const std::string& StringFromLabel(unsigned int label) const {
    auto it = label_to_str_.find(label);
    if (it != label_to_str_.end()) {
      return it->second;
    } else {
      std::cerr << "Invalid label " << label << std::endl;
      abort();
    }
  }

  unsigned int LabelFromString(const std::string& string) const {
    auto it = str_to_label_.find(string);
    if (it != str_to_label_.end()) {
      return it->second;
    } else {
      std::cerr << "Invalid string " << string << std::endl;
      abort();
    }
  }

  size_t GetSize() const {
    return size_;
  }

  bool IsSpace(unsigned int label) const {
    return label == space_label_;
  }

  unsigned int GetSpaceLabel() const {
    return space_label_;
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
  unsigned int space_label_;
  std::unordered_map<unsigned int, std::string> label_to_str_;
  std::unordered_map<std::string, unsigned int> str_to_label_;
};

#endif //ALPHABET_H
