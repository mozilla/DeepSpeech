#ifndef LM_PARTIAL_H
#define LM_PARTIAL_H

#include "lm/return.hh"
#include "lm/state.hh"

#include <algorithm>
#include <cassert>

namespace lm {
namespace ngram {

struct ExtendReturn {
  float adjust;
  bool make_full;
  unsigned char next_use;
};

template <class Model> ExtendReturn ExtendLoop(
    const Model &model,
    unsigned char seen, const WordIndex *add_rbegin, const WordIndex *add_rend, const float *backoff_start,
    const uint64_t *pointers, const uint64_t *pointers_end,
    uint64_t *&pointers_write,
    float *backoff_write) {
  unsigned char add_length = add_rend - add_rbegin;

  float backoff_buf[2][KENLM_MAX_ORDER - 1];
  float *backoff_in = backoff_buf[0], *backoff_out = backoff_buf[1];
  std::copy(backoff_start, backoff_start + add_length, backoff_in);

  ExtendReturn value;
  value.make_full = false;
  value.adjust = 0.0;
  value.next_use = add_length;

  unsigned char i = 0;
  unsigned char length = pointers_end - pointers;
  // pointers_write is NULL means that the existing left state is full, so we should use completed probabilities.
  if (pointers_write) {
    // Using full context, writing to new left state.
    for (; i < length; ++i) {
      FullScoreReturn ret(model.ExtendLeft(
          add_rbegin, add_rbegin + value.next_use,
          backoff_in,
          pointers[i], i + seen + 1,
          backoff_out,
          value.next_use));
      std::swap(backoff_in, backoff_out);
      if (ret.independent_left) {
        value.adjust += ret.prob;
        value.make_full = true;
        ++i;
        break;
      }
      value.adjust += ret.rest;
      *pointers_write++ = ret.extend_left;
      if (value.next_use != add_length) {
        value.make_full = true;
        ++i;
        break;
      }
    }
  }
  // Using some of the new context.
  for (; i < length && value.next_use; ++i) {
    FullScoreReturn ret(model.ExtendLeft(
        add_rbegin, add_rbegin + value.next_use,
        backoff_in,
        pointers[i], i + seen + 1,
        backoff_out,
        value.next_use));
    std::swap(backoff_in, backoff_out);
    value.adjust += ret.prob;
  }
  float unrest = model.UnRest(pointers + i, pointers_end, i + seen + 1);
  // Using none of the new context.
  value.adjust += unrest;

  std::copy(backoff_in, backoff_in + value.next_use, backoff_write);
  return value;
}

template <class Model> float RevealBefore(const Model &model, const Right &reveal, const unsigned char seen, bool reveal_full, Left &left, Right &right) {
  assert(seen < reveal.length || reveal_full);
  uint64_t *pointers_write = reveal_full ? NULL : left.pointers;
  float backoff_buffer[KENLM_MAX_ORDER - 1];
  ExtendReturn value(ExtendLoop(
      model,
      seen, reveal.words + seen, reveal.words + reveal.length, reveal.backoff + seen,
      left.pointers, left.pointers + left.length,
      pointers_write,
      left.full ? backoff_buffer : (right.backoff + right.length)));
  if (reveal_full) {
    left.length = 0;
    value.make_full = true;
  } else {
    left.length = pointers_write - left.pointers;
    value.make_full |= (left.length == model.Order() - 1);
  }
  if (left.full) {
    for (unsigned char i = 0; i < value.next_use; ++i) value.adjust += backoff_buffer[i];
  } else {
    // If left wasn't full when it came in, put words into right state.
    std::copy(reveal.words + seen, reveal.words + seen + value.next_use, right.words + right.length);
    right.length += value.next_use;
    left.full = value.make_full || (right.length == model.Order() - 1);
  }
  return value.adjust;
}

template <class Model> float RevealAfter(const Model &model, Left &left, Right &right, const Left &reveal, unsigned char seen) {
  assert(seen < reveal.length || reveal.full);
  uint64_t *pointers_write = left.full ? NULL : (left.pointers + left.length);
  ExtendReturn value(ExtendLoop(
      model,
      seen, right.words, right.words + right.length, right.backoff,
      reveal.pointers + seen, reveal.pointers + reveal.length,
      pointers_write,
      right.backoff));
  if (reveal.full) {
    for (unsigned char i = 0; i < value.next_use; ++i) value.adjust += right.backoff[i];
    right.length = 0;
    value.make_full = true;
  } else {
    right.length = value.next_use;
    value.make_full |= (right.length == model.Order() - 1);
  }
  if (!left.full) {
    left.length = pointers_write - left.pointers;
    left.full = value.make_full || (left.length == model.Order() - 1);
  }
  return value.adjust;
}

template <class Model> float Subsume(const Model &model, Left &first_left, const Right &first_right, const Left &second_left, Right &second_right, const unsigned int between_length) {
  assert(first_right.length < KENLM_MAX_ORDER);
  assert(second_left.length < KENLM_MAX_ORDER);
  assert(between_length < KENLM_MAX_ORDER - 1);
  uint64_t *pointers_write = first_left.full ? NULL : (first_left.pointers + first_left.length);
  float backoff_buffer[KENLM_MAX_ORDER - 1];
  ExtendReturn value(ExtendLoop(
        model,
        between_length, first_right.words, first_right.words + first_right.length, first_right.backoff,
        second_left.pointers, second_left.pointers + second_left.length,
        pointers_write,
        second_left.full ? backoff_buffer : (second_right.backoff + second_right.length)));
  if (second_left.full) {
    for (unsigned char i = 0; i < value.next_use; ++i) value.adjust += backoff_buffer[i];
  } else {
    std::copy(first_right.words, first_right.words + value.next_use, second_right.words + second_right.length);
    second_right.length += value.next_use;
    value.make_full |= (second_right.length == model.Order() - 1);
  }
  if (!first_left.full) {
    first_left.length = pointers_write - first_left.pointers;
    first_left.full = value.make_full || second_left.full || (first_left.length == model.Order() - 1);
  }
  assert(first_left.length < KENLM_MAX_ORDER);
  assert(second_right.length < KENLM_MAX_ORDER);
  return value.adjust;
}

} // namespace ngram
} // namespace lm

#endif // LM_PARTIAL_H
