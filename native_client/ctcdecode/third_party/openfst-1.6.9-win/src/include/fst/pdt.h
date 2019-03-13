// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Common classes for PDT expansion/traversal.

#ifndef FST_EXTENSIONS_PDT_PDT_H_
#define FST_EXTENSIONS_PDT_PDT_H_

#include <map>
#include <set>
#include <unordered_map>

#include <fst/compat.h>
#include <fst/log.h>
#include <fst/fst.h>
#include <fst/state-table.h>

namespace fst {

// Provides bijection between parenthesis stacks and signed integral stack IDs.
// Each stack ID is unique to each distinct stack. The open-close parenthesis
// label pairs are passed using the parens argument.
template <typename StackId, typename Label>
class PdtStack {
 public:
  // The stacks are stored in a tree. The nodes are stored in a vector. Each
  // node represents the top of some stack and is identified by its position in
  // the vector. Its' parent node represents the stack with the top popped and
  // its children are stored in child_map_ and accessed by stack_id and label.
  // The paren_id is
  // the position in parens of the parenthesis for that node.
  struct StackNode {
    StackId parent_id;
    size_t paren_id;

    StackNode(StackId p, size_t i) : parent_id(p), paren_id(i) {}
  };

  explicit PdtStack(const std::vector<std::pair<Label, Label>> &parens)
      : parens_(parens), min_paren_(kNoLabel), max_paren_(kNoLabel) {
    for (size_t i = 0; i < parens.size(); ++i) {
      const auto &pair = parens[i];
      paren_map_[pair.first] = i;
      paren_map_[pair.second] = i;
      if (min_paren_ == kNoLabel || pair.first < min_paren_) {
        min_paren_ = pair.first;
      }
      if (pair.second < min_paren_) min_paren_ = pair.second;
      if (max_paren_ == kNoLabel || pair.first > max_paren_) {
        max_paren_ = pair.first;
      }
      if (pair.second > max_paren_) max_paren_ = pair.second;
    }
    nodes_.push_back(StackNode(-1, -1));  // Tree root.
  }

  // Returns stack ID given the current stack ID (0 if empty) and label read.
  // Pushes onto the stack if the label is an open parenthesis, returning the
  // new stack ID. Pops the stack if the label is a close parenthesis that
  // matches the top of the stack, returning the parent stack ID. Returns -1 if
  // label is an unmatched close parenthesis. Otherwise, returns the current
  // stack ID.
  StackId Find(StackId stack_id, Label label) {
    if (min_paren_ == kNoLabel || label < min_paren_ || label > max_paren_) {
      return stack_id;  // Non-paren.
    }
    const auto it = paren_map_.find(label);
    // Non-paren.
    if (it == paren_map_.end()) return stack_id;
    const auto paren_id = it->second;
    // Open paren.
    if (label == parens_[paren_id].first) {
      auto &child_id = child_map_[std::make_pair(stack_id, label)];
      if (child_id == 0) {  // Child not found; pushes label.
        child_id = nodes_.size();
        nodes_.push_back(StackNode(stack_id, paren_id));
      }
      return child_id;
    }
    const auto &node = nodes_[stack_id];
    // Matching close paren.
    if (paren_id == node.paren_id) return node.parent_id;
    // Non-matching close paren.
    return -1;
  }

  // Returns the stack ID obtained by popping the label at the top of the
  // current stack ID.
  StackId Pop(StackId stack_id) const { return nodes_[stack_id].parent_id; }

  // Returns the paren ID at the top of the stack.
  std::ptrdiff_t Top(StackId stack_id) const { return nodes_[stack_id].paren_id; }

  std::ptrdiff_t ParenId(Label label) const {
    const auto it = paren_map_.find(label);
    if (it == paren_map_.end()) return -1;  // Non-paren.
    return it->second;
  }

 private:
  struct ChildHash {
    size_t operator()(const std::pair<StackId, Label> &pair) const {
      static constexpr size_t prime = 7853;
      return static_cast<size_t>(pair.first) +
             static_cast<size_t>(pair.second) * prime;
    }
  };

  std::vector<std::pair<Label, Label>> parens_;
  std::vector<StackNode> nodes_;
  std::unordered_map<Label, size_t> paren_map_;
  // Child of stack node w.r.t label
  std::unordered_map<std::pair<StackId, Label>, StackId, ChildHash> child_map_;
  Label min_paren_;
  Label max_paren_;
};

// State tuple for PDT expansion.
template <typename S, typename K>
struct PdtStateTuple {
  using StateId = S;
  using StackId = K;

  StateId state_id;
  StackId stack_id;

  PdtStateTuple(StateId state_id = kNoStateId, StackId stack_id = -1)
      : state_id(state_id), stack_id(stack_id) {}
};

// Equality of PDT state tuples.
template <typename S, typename K>
inline bool operator==(const PdtStateTuple<S, K> &x,
                       const PdtStateTuple<S, K> &y) {
  if (&x == &y) return true;
  return x.state_id == y.state_id && x.stack_id == y.stack_id;
}

// Hash function object for PDT state tuples
template <class T>
class PdtStateHash {
 public:
  size_t operator()(const T &tuple) const {
    static constexpr auto prime = 7853;
    return tuple.state_id + tuple.stack_id * prime;
  }
};

// Tuple to PDT state bijection.
template <class StateId, class StackId>
class PdtStateTable : public CompactHashStateTable<
                          PdtStateTuple<StateId, StackId>,
                          PdtStateHash<PdtStateTuple<StateId, StackId>>> {
 public:
  PdtStateTable() {}

  PdtStateTable(const PdtStateTable &other) {}

 private:
  PdtStateTable &operator=(const PdtStateTable &) = delete;
};

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_PDT_H_
