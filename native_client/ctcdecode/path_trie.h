#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "alphabet.h"
#include "object_pool.h"

/* Tree structure with parent and children information
 * It is used to store the timesteps data for the PathTrie below
 */
template<class DataT>
struct TreeNode{
	std::shared_ptr<TreeNode<DataT>> parent;
    std::vector<std::shared_ptr<TreeNode<DataT>>> children;

    DataT data;

    TreeNode(std::shared_ptr<TreeNode<DataT>> const& parent_, DataT const& data_): parent{parent_}, data{data_} {}
};

template<class NodeDataT, class ChildDataT>
std::shared_ptr<TreeNode<NodeDataT>> add_child(std::shared_ptr<TreeNode<NodeDataT>> const& node, ChildDataT&& data_);

template<class DataT>
std::vector<DataT> get_history(TreeNode<DataT>*);

using TimestepTreeNode=TreeNode<unsigned int>;

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
public:
  using FstType = fst::ConstFst<fst::StdArc>;

  PathTrie();
  ~PathTrie();

  // get new prefix after appending new char
  PathTrie* get_path_trie(unsigned int new_char, float log_prob_c, bool reset = true);

  // get the prefix data in correct time order from root to current node
  void get_path_vec(std::vector<unsigned int>& output);

  // get the prefix data in correct time order from beginning of last grapheme to current node
  PathTrie* get_prev_grapheme(std::vector<unsigned int>& output,
                              const Alphabet& alphabet);

  // get the distance from current node to the first codepoint boundary, and the byte value at the boundary
  int distance_to_codepoint_boundary(unsigned char *first_byte, const Alphabet& alphabet);

  // get the prefix data in correct time order from beginning of last word to current node
  PathTrie* get_prev_word(std::vector<unsigned int>& output,
                          const Alphabet& alphabet);

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
  unsigned int character;
  std::shared_ptr<TimestepTreeNode> timesteps;

  // timestep temporary storage for each decoding step. 
  std::shared_ptr<TimestepTreeNode> previous_timesteps=nullptr; 
  unsigned int new_timestep;

  PathTrie* parent;

private:
  int ROOT_;
  bool exists_;
  bool has_dictionary_;

  std::vector<std::pair<unsigned int, PathTrie*>> children_;

  // pointer to dictionary of FST
  std::shared_ptr<FstType> dictionary_;
  FstType::StateId dictionary_state_;
  std::shared_ptr<fst::SortedMatcher<FstType>> matcher_;
};

// TreeNode implementation
template<class NodeDataT, class ChildDataT>
std::shared_ptr<TreeNode<NodeDataT>> add_child(std::shared_ptr<TreeNode<NodeDataT>> const& node, ChildDataT&& data_){
	static godefv::memory::object_pool_t<TreeNode<NodeDataT>> tree_node_pool;
	node->children.emplace_back(tree_node_pool.make_unique(node, std::forward<ChildDataT>(data_)));
    return node->children.back();
}

template<class DataT>
void get_history_helper(std::shared_ptr<TreeNode<DataT>> const& tree_node, std::vector<DataT>* output){
    if(tree_node==nullptr) return;
	assert(tree_node->parent != tree_node);
    get_history_helper(tree_node->parent, output);
    output->push_back(tree_node->data);
}
template<class DataT>
std::vector<DataT> get_history(std::shared_ptr<TreeNode<DataT>> const& tree_node){
    std::vector<DataT> output;
    get_history_helper(tree_node, &output);
    return output;
}


#endif  // PATH_TRIE_H
