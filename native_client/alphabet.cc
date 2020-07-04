#include "alphabet.h"
#include "ctcdecode/decoder_utils.h"

#include <fstream>

// std::getline, but handle newline conventions from multiple platforms instead
// of just the platform this code was built for
std::istream&
getline_crossplatform(std::istream& is, std::string& t)
{
  t.clear();

  // The characters in the stream are read one-by-one using a std::streambuf.
  // That is faster than reading them one-by-one using the std::istream.
  // Code that uses streambuf this way must be guarded by a sentry object.
  // The sentry object performs various tasks,
  // such as thread synchronization and updating the stream state.
  std::istream::sentry se(is, true);
  std::streambuf* sb = is.rdbuf();

  while (true) {
    int c = sb->sbumpc();
    switch (c) {
    case '\n':
      return is;
    case '\r':
      if(sb->sgetc() == '\n')
          sb->sbumpc();
      return is;
    case std::streambuf::traits_type::eof():
      // Also handle the case when the last line has no line ending
      if(t.empty())
        is.setstate(std::ios::eofbit);
      return is;
    default:
      t += (char)c;
    }
  }
}

int
Alphabet::init(const char *config_file)
{
  std::ifstream in(config_file, std::ios::in);
  if (!in) {
    return 1;
  }
  unsigned int label = 0;
  space_label_ = -2;
  for (std::string line; getline_crossplatform(in, line);) {
    if (line.size() == 2 && line[0] == '\\' && line[1] == '#') {
      line = '#';
    } else if (line[0] == '#') {
      continue;
    }
    //TODO: we should probably do something more i18n-aware here
    if (line == " ") {
      space_label_ = label;
    }
    if (line.length() == 0) {
      continue;
    }
    label_to_str_[label] = line;
    str_to_label_[line] = label;
    ++label;
  }
  size_ = label;
  in.close();
  return 0;
}

std::string
Alphabet::Serialize()
{
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

int
Alphabet::Deserialize(const char* buffer, const int buffer_size)
{
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

std::string
Alphabet::DecodeSingle(unsigned int label) const
{
  auto it = label_to_str_.find(label);
  if (it != label_to_str_.end()) {
    return it->second;
  } else {
    std::cerr << "Invalid label " << label << std::endl;
    abort();
  }
}

unsigned int
Alphabet::EncodeSingle(const std::string& string) const
{
  auto it = str_to_label_.find(string);
  if (it != str_to_label_.end()) {
    return it->second;
  } else {
    std::cerr << "Invalid string " << string << std::endl;
    abort();
  }
}

std::string
Alphabet::Decode(const std::vector<unsigned int>& input) const
{
  std::string word;
  for (auto ind : input) {
    word += DecodeSingle(ind);
  }
  return word;
}

std::string
Alphabet::Decode(const unsigned int* input, int length) const
{
  std::string word;
  for (int i = 0; i < length; ++i) {
    word += DecodeSingle(input[i]);
  }
  return word;
}

std::vector<unsigned int>
Alphabet::Encode(const std::string& input) const
{
  std::vector<unsigned int> result;
  for (auto cp : split_into_codepoints(input)) {
    result.push_back(EncodeSingle(cp));
  }
  return result;
}

std::vector<unsigned int>
UTF8Alphabet::Encode(const std::string& input) const
{
  std::vector<unsigned int> result;
  for (auto byte_char : input) {
    std::string byte_str(1, byte_char);
    result.push_back(EncodeSingle(byte_str));
  }
  return result;
}
