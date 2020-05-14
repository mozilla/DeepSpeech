#include <algorithm>
#include <iostream>
#include <string>

#include "ctcdecode/scorer.h"
#include "alphabet.h"

#ifdef DEBUG
#include <limits>
#include <unordered_map>
#include "ctcdecode/path_trie.h"
#endif // DEBUG

using namespace std;

int main(int argc, char** argv)
{
  const char* kenlm_path    = argv[1];
  const char* trie_path     = argv[2];
  const char* alphabet_path = argv[3];

  printf("Loading trie(%s) and alphabet(%s)\n", trie_path, alphabet_path);

  Alphabet alphabet;
  int err = alphabet.init(alphabet_path);
  if (err != 0) {
    return err;
  }
  Scorer scorer;
  err = scorer.init(kenlm_path, alphabet);
#ifndef DEBUG
  return err;
#else
  // Print some info about the FST
  using FstType = fst::ConstFst<fst::StdArc>;

  auto dict = scorer.dictionary.get();

  struct state_info {
    int range_min = std::numeric_limits<int>::max();
    int range_max = std::numeric_limits<int>::min();
  };

  auto print_states_from = [&](int i) {
    std::unordered_map<int, state_info> sinfo;
    for (fst::ArcIterator<FstType> aiter(*dict, i); !aiter.Done(); aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      sinfo[arc.nextstate].range_min = std::min(sinfo[arc.nextstate].range_min, arc.ilabel-1);
      sinfo[arc.nextstate].range_max = std::max(sinfo[arc.nextstate].range_max, arc.ilabel-1);
    }

    for (auto it = sinfo.begin(); it != sinfo.end(); ++it) {
      state_info s = it->second;
      printf("%d -> state %d (chars 0x%X - 0x%X, '%c' - '%c')\n", i, it->first, (unsigned int)s.range_min, (unsigned int)s.range_max, (char)s.range_min, (char)s.range_max);
    }
  };

  print_states_from(0);

  // for (int i = 1; i < 10; ++i) {
  //   print_states_from(i);
  // }
  return 0;
#endif // DEBUG
}
