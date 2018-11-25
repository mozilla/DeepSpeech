// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Compresses and decompresses unweighted FSTs.

#ifndef FST_EXTENSIONS_COMPRESS_ELIAS_H_
#define FST_EXTENSIONS_COMPRESS_ELIAS_H_

#include <stack>
#include <vector>

#include <fst/compat.h>
namespace fst {

template <class Var>
class Elias {
 public:
  // Gamma encoding is a subroutine for Delta encoding
  static void GammaEncode(const Var &input, std::vector<bool> *code);

  // Elias Delta encoding for a single integer
  static void DeltaEncode(const Var &input, std::vector<bool> *code);

  // Batch decoding of a set of integers
  static void BatchDecode(const std::vector<bool> &input,
                          std::vector<Var> *output);
};

template <class Var>
void Elias<Var>::GammaEncode(const Var &input, std::vector<bool> *code) {
  Var input_copy = input;
  std::stack<bool> reverse_code;
  while (input_copy > 0) {
    reverse_code.push(input_copy % 2);
    input_copy = input_copy / 2;
  }
  for (Var auxvar = 0; auxvar < reverse_code.size() - 1; auxvar++)
    code->push_back(0);
  while (reverse_code.empty() != 1) {
    code->push_back(reverse_code.top());
    reverse_code.pop();
  }
}
template <class Var>
void Elias<Var>::DeltaEncode(const Var &input, std::vector<bool> *code) {
  Var input_copy = input + 1;
  std::stack<bool> reverse_remainder;
  Var auxvar = 0;
  while (input_copy != 0) {
    reverse_remainder.push(input_copy % 2);
    input_copy = input_copy / 2;
    auxvar = auxvar + 1;
  }
  GammaEncode(auxvar, code);
  reverse_remainder.pop();
  while (reverse_remainder.empty() != 1) {
    code->push_back(reverse_remainder.top());
    reverse_remainder.pop();
  }
}

template <class Var>
void Elias<Var>::BatchDecode(const std::vector<bool> &input,
                             std::vector<Var> *output) {
  Var lead_zeros = 0;
  Var remainder_bits = 0;
  Var current_word = 1;
  Var value = 1;
  std::vector<bool>::const_iterator it = input.begin();
  while (it != input.end()) {
    lead_zeros = 0;
    remainder_bits = 0;
    current_word = 1;
    value = 1;
    while (*it != 1) {
      it++;
      lead_zeros++;
    }
    it++;
    while (lead_zeros > 0) {
      lead_zeros--;
      current_word = 2 * current_word + *it;
      it++;
    }
    current_word--;
    while (current_word > 0) {
      value = 2 * value + *it;
      current_word--;
      it++;
    }
    output->push_back(value - 1);
  }
}

}  // namespace fst

#endif  // FST_EXTENSIONS_COMPRESS_ELIAS_H_
