// Separate header because this is used often.
#ifndef LM_WORD_INDEX_H
#define LM_WORD_INDEX_H

#include <climits>

namespace lm {
typedef unsigned int WordIndex;
const WordIndex kMaxWordIndex = UINT_MAX;
const WordIndex kUNK = 0;
} // namespace lm

typedef lm::WordIndex LMWordIndex;

#endif
