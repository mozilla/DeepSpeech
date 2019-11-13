#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "fst/fstlib.h"

#ifdef DEBUG
#include "alphabet.h"
#endif

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
public:
  using FstType = fst::ConstFst<fst::StdArc>;

  PathTrie();
  ~PathTrie();

  // get new prefix after appending new char
  PathTrie* get_path_trie(int new_char, int new_timestep, float log_prob_c, bool reset = true);

  // get the prefix data in correct time order from root to current node
  void get_path_vec(std::vector<int>& output, std::vector<int>& timesteps);

  // get the prefix data in correct time order from beginning of last grapheme to current node
  PathTrie* get_prev_grapheme(std::vector<int>& output,
                              std::vector<int>& timesteps);

  // get the distance from current node to the first codepoint boundary, and the byte value at the boundary
  int distance_to_codepoint_boundary(unsigned char *first_byte);

  // get the prefix data in correct time order from beginning of last word to current node
  PathTrie* get_prev_word(std::vector<int>& output,
                          std::vector<int>& timesteps,
                          int space_id);

  // update log probs
  void iterate_to_vec(std::vector<PathTrie*>& output);

  // set dictionary for FST
  void set_dictionary(std::shared_ptr<FstType> dictionary);

  void set_matcher(std::shared_ptr<fst::SortedMatcher<FstType>>);

  bool is_empty() { return ROOT_ == character; }

  // remove current path from root
  void remove();

#ifdef DEBUG
  void vec(std::vector<PathTrie*>& out);
  void print(const Alphabet& a);
#endif // DEBUG

  float log_prob_b_prev;
  float log_prob_nb_prev;
  float log_prob_b_cur;
  float log_prob_nb_cur;
  float log_prob_c;
  float score;
  float approx_ctc;
  int character;
  int timestep;
  PathTrie* parent;

private:
  int ROOT_;
  bool exists_;
  bool has_dictionary_;

  std::vector<std::pair<int, PathTrie*>> children_;

  // pointer to dictionary of FST
  std::shared_ptr<FstType> dictionary_;
  FstType::StateId dictionary_state_;
  // true if finding ars in FST
  std::shared_ptr<fst::SortedMatcher<FstType>> matcher_;
};

#endif  // PATH_TRIE_H
