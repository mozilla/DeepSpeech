// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_LINEAR_TRIE_H_
#define FST_EXTENSIONS_LINEAR_TRIE_H_

#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/compat.h>
#include <fst/util.h>

namespace fst {

const int kNoTrieNodeId = -1;

// Forward declarations of all available trie topologies.
template <class L, class H>
class NestedTrieTopology;
template <class L, class H>
class FlatTrieTopology;

// A pair of parent node id and label, part of a trie edge
template <class L>
struct ParentLabel {
  int parent;
  L label;

  ParentLabel() {}
  ParentLabel(int p, L l) : parent(p), label(l) {}

  bool operator==(const ParentLabel &that) const {
    return parent == that.parent && label == that.label;
  }

  std::istream &Read(std::istream &strm) {  // NOLINT
    ReadType(strm, &parent);
    ReadType(strm, &label);
    return strm;
  }

  std::ostream &Write(std::ostream &strm) const {  // NOLINT
    WriteType(strm, parent);
    WriteType(strm, label);
    return strm;
  }
};

template <class L, class H>
struct ParentLabelHash {
  size_t operator()(const ParentLabel<L> &pl) const {
    return static_cast<size_t>(pl.parent * 7853 + H()(pl.label));
  }
};

// The trie topology in a nested tree of hash maps; allows efficient
// iteration over children of a specific node.
template <class L, class H>
class NestedTrieTopology {
 public:
  typedef L Label;
  typedef H Hash;
  typedef std::unordered_map<L, int, H> NextMap;

  class const_iterator {
   public:
    typedef std::forward_iterator_tag iterator_category;
    typedef std::pair<ParentLabel<L>, int> value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const value_type *pointer;
    typedef const value_type &reference;

    friend class NestedTrieTopology<L, H>;

    const_iterator() : ptr_(nullptr), cur_node_(kNoTrieNodeId), cur_edge_() {}

    reference operator*() {
      UpdateStub();
      return stub_;
    }
    pointer operator->() {
      UpdateStub();
      return &stub_;
    }

    const_iterator &operator++();
    const_iterator &operator++(int);  // NOLINT

    bool operator==(const const_iterator &that) const {
      return ptr_ == that.ptr_ && cur_node_ == that.cur_node_ &&
             cur_edge_ == that.cur_edge_;
    }
    bool operator!=(const const_iterator &that) const {
      return !(*this == that);
    }

   private:
    const_iterator(const NestedTrieTopology *ptr, int cur_node)
        : ptr_(ptr), cur_node_(cur_node) {
      SetProperCurEdge();
    }

    void SetProperCurEdge() {
      if (cur_node_ < ptr_->NumNodes())
        cur_edge_ = ptr_->nodes_[cur_node_]->begin();
      else
        cur_edge_ = ptr_->nodes_[0]->begin();
    }

    void UpdateStub() {
      stub_.first = ParentLabel<L>(cur_node_, cur_edge_->first);
      stub_.second = cur_edge_->second;
    }

    const NestedTrieTopology *ptr_;
    int cur_node_;
    typename NextMap::const_iterator cur_edge_;
    value_type stub_;
  };

  NestedTrieTopology();
  NestedTrieTopology(const NestedTrieTopology &that);
  ~NestedTrieTopology();
  void swap(NestedTrieTopology &that);
  NestedTrieTopology &operator=(const NestedTrieTopology &that);
  bool operator==(const NestedTrieTopology &that) const;
  bool operator!=(const NestedTrieTopology &that) const;

  int Root() const { return 0; }
  size_t NumNodes() const { return nodes_.size(); }
  int Insert(int parent, const L &label);
  int Find(int parent, const L &label) const;
  const NextMap &ChildrenOf(int parent) const { return *nodes_[parent]; }

  std::istream &Read(std::istream &strm);         // NOLINT
  std::ostream &Write(std::ostream &strm) const;  // NOLINT

  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, NumNodes()); }

 private:
  std::vector<NextMap *>
      nodes_;  // Use pointers to avoid copying the maps when the
               // vector grows
};

template <class L, class H>
NestedTrieTopology<L, H>::NestedTrieTopology() {
  nodes_.push_back(new NextMap);
}

template <class L, class H>
NestedTrieTopology<L, H>::NestedTrieTopology(const NestedTrieTopology &that) {
  nodes_.reserve(that.nodes_.size());
  for (size_t i = 0; i < that.nodes_.size(); ++i) {
    NextMap *node = that.nodes_[i];
    nodes_.push_back(new NextMap(*node));
  }
}

template <class L, class H>
NestedTrieTopology<L, H>::~NestedTrieTopology() {
  for (size_t i = 0; i < nodes_.size(); ++i) {
    NextMap *node = nodes_[i];
    delete node;
  }
}

// TODO(wuke): std::swap compatibility
template <class L, class H>
inline void NestedTrieTopology<L, H>::swap(NestedTrieTopology &that) {
  nodes_.swap(that.nodes_);
}

template <class L, class H>
inline NestedTrieTopology<L, H> &NestedTrieTopology<L, H>::operator=(
    const NestedTrieTopology &that) {
  NestedTrieTopology copy(that);
  swap(copy);
  return *this;
}

template <class L, class H>
inline bool NestedTrieTopology<L, H>::operator==(
    const NestedTrieTopology &that) const {
  if (NumNodes() != that.NumNodes()) return false;
  for (int i = 0; i < NumNodes(); ++i)
    if (ChildrenOf(i) != that.ChildrenOf(i)) return false;
  return true;
}

template <class L, class H>
inline bool NestedTrieTopology<L, H>::operator!=(
    const NestedTrieTopology &that) const {
  return !(*this == that);
}

template <class L, class H>
inline int NestedTrieTopology<L, H>::Insert(int parent, const L &label) {
  int ret = Find(parent, label);
  if (ret == kNoTrieNodeId) {
    ret = NumNodes();
    (*nodes_[parent])[label] = ret;
    nodes_.push_back(new NextMap);
  }
  return ret;
}

template <class L, class H>
inline int NestedTrieTopology<L, H>::Find(int parent, const L &label) const {
  typename NextMap::const_iterator it = nodes_[parent]->find(label);
  return it == nodes_[parent]->end() ? kNoTrieNodeId : it->second;
}

template <class L, class H>
inline std::istream &NestedTrieTopology<L, H>::Read(
    std::istream &strm) {  // NOLINT
  NestedTrieTopology new_trie;
  size_t num_nodes;
  if (!ReadType(strm, &num_nodes)) return strm;
  for (size_t i = 1; i < num_nodes; ++i) new_trie.nodes_.push_back(new NextMap);
  for (size_t i = 0; i < num_nodes; ++i) ReadType(strm, new_trie.nodes_[i]);
  if (strm) swap(new_trie);
  return strm;
}

template <class L, class H>
inline std::ostream &NestedTrieTopology<L, H>::Write(
    std::ostream &strm) const {  // NOLINT
  WriteType(strm, NumNodes());
  for (size_t i = 0; i < NumNodes(); ++i) WriteType(strm, *nodes_[i]);
  return strm;
}

template <class L, class H>
inline typename NestedTrieTopology<L, H>::const_iterator
    &NestedTrieTopology<L, H>::const_iterator::operator++() {
  ++cur_edge_;
  if (cur_edge_ == ptr_->nodes_[cur_node_]->end()) {
    ++cur_node_;
    while (cur_node_ < ptr_->NumNodes() && ptr_->nodes_[cur_node_]->empty())
      ++cur_node_;
    SetProperCurEdge();
  }
  return *this;
}

template <class L, class H>
inline typename NestedTrieTopology<L, H>::const_iterator
    &NestedTrieTopology<L, H>::const_iterator::operator++(int) {  // NOLINT
  const_iterator save(*this);
  ++(*this);
  return save;
}

// The trie topology in a single hash map; only allows iteration over
// all the edges in arbitrary order.
template <class L, class H>
class FlatTrieTopology {
 private:
  typedef std::unordered_map<ParentLabel<L>, int, ParentLabelHash<L, H>>
      NextMap;

 public:
  // Iterator over edges as std::pair<ParentLabel<L>, int>
  typedef typename NextMap::const_iterator const_iterator;
  typedef L Label;
  typedef H Hash;

  FlatTrieTopology() {}
  FlatTrieTopology(const FlatTrieTopology &that) : next_(that.next_) {}
  template <class T>
  explicit FlatTrieTopology(const T &that);

  // TODO(wuke): std::swap compatibility
  void swap(FlatTrieTopology &that) { next_.swap(that.next_); }

  bool operator==(const FlatTrieTopology &that) const {
    return next_ == that.next_;
  }
  bool operator!=(const FlatTrieTopology &that) const {
    return !(*this == that);
  }

  int Root() const { return 0; }
  size_t NumNodes() const { return next_.size() + 1; }
  int Insert(int parent, const L &label);
  int Find(int parent, const L &label) const;

  std::istream &Read(std::istream &strm) {  // NOLINT
    return ReadType(strm, &next_);
  }
  std::ostream &Write(std::ostream &strm) const {  // NOLINT
    return WriteType(strm, next_);
  }

  const_iterator begin() const { return next_.begin(); }
  const_iterator end() const { return next_.end(); }

 private:
  NextMap next_;
};

template <class L, class H>
template <class T>
FlatTrieTopology<L, H>::FlatTrieTopology(const T &that)
    : next_(that.begin(), that.end()) {}

template <class L, class H>
inline int FlatTrieTopology<L, H>::Insert(int parent, const L &label) {
  int ret = Find(parent, label);
  if (ret == kNoTrieNodeId) {
    ret = NumNodes();
    next_[ParentLabel<L>(parent, label)] = ret;
  }
  return ret;
}

template <class L, class H>
inline int FlatTrieTopology<L, H>::Find(int parent, const L &label) const {
  typename NextMap::const_iterator it =
      next_.find(ParentLabel<L>(parent, label));
  return it == next_.end() ? kNoTrieNodeId : it->second;
}

// A collection of implementations of the trie data structure. The key
// is a sequence of type `L` which must be hashable. The value is of
// `V` which must be default constructible and copyable. In addition,
// a value object is stored for each node in the trie therefore
// copying `V` should be cheap.
//
// One can access the store values with an integer node id, using the
// [] operator. A valid node id can be obtained by the following ways:
//
// 1. Using the `Root()` method to get the node id of the root.
//
// 2. Iterating through 0 to `NumNodes() - 1`. The node ids are dense
// so every integer in this range is a valid node id.
//
// 3. Using the node id returned from a successful `Insert()` or
// `Find()` call.
//
// 4. Iterating over the trie edges with an `EdgeIterator` and using
// the node ids returned from its `Parent()` and `Child()` methods.
//
// Below is an example of inserting keys into the trie:
//
//   const string words[] = {"hello", "health", "jello"};
//   Trie<char, bool> dict;
//   for (auto word : words) {
//     int cur = dict.Root();
//     for (char c : word) {
//       cur = dict.Insert(cur, c);
//     }
//     dict[cur] = true;
//   }
//
// And the following is an example of looking up the longest prefix of
// a string using the trie constructed above:
//
//   string query = "healed";
//   size_t prefix_length = 0;
//   int cur = dict.Find(dict.Root(), query[prefix_length]);
//   while (prefix_length < query.size() &&
//     cur != Trie<char, bool>::kNoNodeId) {
//     ++prefix_length;
//     cur = dict.Find(cur, query[prefix_length]);
//   }
template <class L, class V, class T>
class MutableTrie {
 public:
  template <class LL, class VV, class TT>
  friend class MutableTrie;

  typedef L Label;
  typedef V Value;
  typedef T Topology;

  // Constructs a trie with only the root node.
  MutableTrie() {}

  // Conversion from another trie of a possiblly different
  // topology. The underlying topology must supported conversion.
  template <class S>
  explicit MutableTrie(const MutableTrie<L, V, S> &that)
      : topology_(that.topology_), values_(that.values_) {}

  // TODO(wuke): std::swap compatibility
  void swap(MutableTrie &that) {
    topology_.swap(that.topology_);
    values_.swap(that.values_);
  }

  int Root() const { return topology_.Root(); }
  size_t NumNodes() const { return topology_.NumNodes(); }

  // Inserts an edge with given `label` at node `parent`. Returns the
  // child node id. If the node already exists, returns the node id
  // right away.
  int Insert(int parent, const L &label) {
    int ret = topology_.Insert(parent, label);
    values_.resize(NumNodes());
    return ret;
  }

  // Finds the node id of the node from `parent` via `label`. Returns
  // `kNoTrieNodeId` when such a node does not exist.
  int Find(int parent, const L &label) const {
    return topology_.Find(parent, label);
  }

  const T &TrieTopology() const { return topology_; }

  // Accesses the value stored for the given node.
  V &operator[](int node_id) { return values_[node_id]; }
  const V &operator[](int node_id) const { return values_[node_id]; }

  // Comparison by content
  bool operator==(const MutableTrie &that) const {
    return topology_ == that.topology_ && values_ == that.values_;
  }

  bool operator!=(const MutableTrie &that) const { return !(*this == that); }

  std::istream &Read(std::istream &strm) {  // NOLINT
    ReadType(strm, &topology_);
    ReadType(strm, &values_);
    return strm;
  }
  std::ostream &Write(std::ostream &strm) const {  // NOLINT
    WriteType(strm, topology_);
    WriteType(strm, values_);
    return strm;
  }

 private:
  T topology_;
  std::vector<V> values_;
};

}  // namespace fst

#endif  // FST_EXTENSIONS_LINEAR_TRIE_H_
