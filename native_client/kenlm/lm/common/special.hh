#ifndef LM_COMMON_SPECIAL_H
#define LM_COMMON_SPECIAL_H

#include "lm/word_index.hh"

namespace lm {

class SpecialVocab {
  public:
    SpecialVocab(WordIndex bos, WordIndex eos) : bos_(bos), eos_(eos) {}

    bool IsSpecial(WordIndex word) const {
      return word == kUNK || word == bos_ || word == eos_;
    }

    WordIndex UNK() const { return kUNK; }
    WordIndex BOS() const { return bos_; }
    WordIndex EOS() const { return eos_; }

  private:
    WordIndex bos_;
    WordIndex eos_;
};

} // namespace lm

#endif // LM_COMMON_SPECIAL_H
