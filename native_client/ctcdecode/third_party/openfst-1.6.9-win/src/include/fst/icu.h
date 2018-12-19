// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// This library implements an unrestricted Thompson/Pike UTF-8 parser and
// serializer. UTF-8 is a restricted subset of this byte stream encoding. For
// a description of the encoding details, see:
//
//     http://en.wikipedia.org/wiki/UTF-8

#ifndef FST_ICU_H_
#define FST_ICU_H_

#include <sstream>
#include <vector>

#include <fst/log.h>

namespace fst {

// This function writes UTF-8 strings into a vector of Labels, truncating if
// necessary. It is possible to use this sensibly with as little as 16 bits of
// Label precision (i.e., when all characters are within the Basic Multilingual
// Plane). With 21 bits, one can label all UTF-8 labelpoints, including those
// from the various Astral Planes. Naturally, it is safe to use this with larger
// Labels (e.g., 64 bits).
template <class Label>
bool UTF8StringToLabels(const string &str, std::vector<Label> *labels) {
  for (auto it = str.begin(); it != str.end();) {
    int c = *it & 0xff;
    ++it;
    if ((c & 0x80) == 0) {
      labels->emplace_back(c);
    } else {
      if ((c & 0xc0) == 0x80) {
        LOG(ERROR) << "UTF8StringToLabels: Continuation byte as lead byte";
        return false;
      }
      int count =
          (c >= 0xc0) + (c >= 0xe0) + (c >= 0xf0) + (c >= 0xf8) + (c >= 0xfc);
      int32_t label = c & ((1 << (6 - count)) - 1);
      while (count != 0) {
        if (it == str.end()) {
          LOG(ERROR) << "UTF8StringToLabels: Truncated UTF-8 byte sequence";
          return false;
        }
        char cb = *it;
        ++it;
        if ((cb & 0xc0) != 0x80) {
          LOG(ERROR) << "UTF8StringToLabels: Missing/invalid continuation byte";
          return false;
        }
        label = (label << 6) | (cb & 0x3f);
        --count;
      }
      if (label < 0) {
        // Should be unreachable.
        LOG(ERROR) << "UTF8StringToLabels: Invalid character found: " << c;
        return false;
      }
      labels->push_back(label);
    }
  }
  return true;
}

template <class Label>
bool LabelsToByteString(const std::vector<Label> &labels, string *str) {
  std::ostringstream ostrm;
  for (const char label : labels) {
    if (label != 0) ostrm << label;
  }
  *str = ostrm.str();
  return !!ostrm;
}

template <class Label>
bool LabelsToUTF8String(const std::vector<Label> &labels, string *str) {
  std::ostringstream ostrm;
  for (const int32_t label : labels) {
    if (label < 0) {
      LOG(ERROR) << "LabelsToUTF8String: Invalid character found: " << label;
      return false;
    } else if (label == 0) {
      continue;
    } else if (label < 0x80) {
      ostrm << static_cast<char>(label);
    } else if (label < 0x800) {
      ostrm << static_cast<char>((label >> 6) | 0xc0);
      ostrm << static_cast<char>((label & 0x3f) | 0x80);
    } else if (label < 0x10000) {
      ostrm << static_cast<char>((label >> 12) | 0xe0);
      ostrm << static_cast<char>(((label >> 6) & 0x3f) | 0x80);
      ostrm << static_cast<char>((label & 0x3f) | 0x80);
    } else if (label < 0x200000) {
      ostrm << static_cast<char>((label >> 18) | 0xf0);
      ostrm << static_cast<char>(((label >> 12) & 0x3f) | 0x80);
      ostrm << static_cast<char>(((label >> 6) & 0x3f) | 0x80);
      ostrm << static_cast<char>((label & 0x3f) | 0x80);
    } else if (label < 0x4000000) {
      ostrm << static_cast<char>((label >> 24) | 0xf8);
      ostrm << static_cast<char>(((label >> 18) & 0x3f) | 0x80);
      ostrm << static_cast<char>(((label >> 12) & 0x3f) | 0x80);
      ostrm << static_cast<char>(((label >> 6) & 0x3f) | 0x80);
      ostrm << static_cast<char>((label & 0x3f) | 0x80);
    } else {
      ostrm << static_cast<char>((label >> 30) | 0xfc);
      ostrm << static_cast<char>(((label >> 24) & 0x3f) | 0x80);
      ostrm << static_cast<char>(((label >> 18) & 0x3f) | 0x80);
      ostrm << static_cast<char>(((label >> 12) & 0x3f) | 0x80);
      ostrm << static_cast<char>(((label >> 6) & 0x3f) | 0x80);
      ostrm << static_cast<char>((label & 0x3f) | 0x80);
    }
  }
  *str = ostrm.str();
  return !!ostrm;
}

}  // namespace fst

#endif  // FST_ICU_H_
