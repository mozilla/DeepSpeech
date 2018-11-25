#include "lm/virtual_interface.hh"

#include "lm/lm_exception.hh"

namespace lm {
namespace base {

Vocabulary::~Vocabulary() {}

void Vocabulary::SetSpecial(WordIndex begin_sentence, WordIndex end_sentence, WordIndex not_found) {
  begin_sentence_ = begin_sentence;
  end_sentence_ = end_sentence;
  not_found_ = not_found;
}

Model::~Model() {}

} // namespace base
} // namespace lm
