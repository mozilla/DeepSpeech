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

  virtual int init(const char *config_file) {
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

  std::string serialize() {
    // Serialization format is a sequence of (key, value) pairs, where key is
    // a uint16_t and value is a uint16_t length followed by `length` UTF-8
    // encoded bytes with the label.
    std::stringstream out;

    // We start by writing the number of pairs in the buffer as uint16_t.
    uint16_t size = size_;
    out.write(reinterpret_cast<char*>(&size), sizeof(size));

    for (auto it = label_to_str_.begin(); it != label_to_str_.end(); ++it) {
      uint16_t key = it->first;
      string str = it->second;
      uint16_t len = str.length();
      // Then we write the key as uint16_t, followed by the length of the value
      // as uint16_t, followed by `length` bytes (the value itself).
      out.write(reinterpret_cast<char*>(&key), sizeof(key));
      out.write(reinterpret_cast<char*>(&len), sizeof(len));
      out.write(str.data(), len);
    }

    return out.str();
  }

  int deserialize(const char* buffer, const int buffer_size) {
    // See util/text.py for an explanation of the serialization format.
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

protected:
  size_t size_;
  unsigned int space_label_;
  std::unordered_map<unsigned int, std::string> label_to_str_;
  std::unordered_map<std::string, unsigned int> str_to_label_;
};

class UTF8Alphabet : public Alphabet
{
public:
  UTF8Alphabet() {
    size_ = 255;
    space_label_ = ' ' - 1;
    for (int i = 0; i < size_; ++i) {
      std::string val(1, i+1);
      label_to_str_[i] = val;
      str_to_label_[val] = i;
    }
  }

  int init(const char*) override {}
};


#endif //ALPHABET_H
