#ifndef DECODERSTATE_H_
#define DECODERSTATE_H_

#include <vector>

/* Struct for the state of the decoder, containing the prefixes and initial root prefix plus state variables. */

struct DecoderState {
  int time_step;
  int space_id;
  int blank_id;
  std::vector<PathTrie*> prefixes;
  PathTrie *prefix_root;
  
  ~DecoderState() {
    if (prefix_root != nullptr) {
      delete prefix_root;
    }
    prefix_root = nullptr;
  }
};

#endif  // DECODERSTATE_H_
