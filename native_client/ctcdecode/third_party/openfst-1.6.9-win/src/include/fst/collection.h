// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to store a collection of ordered (multi-)sets with elements of type T.

#ifndef FST_EXTENSIONS_PDT_COLLECTION_H_
#define FST_EXTENSIONS_PDT_COLLECTION_H_

#include <functional>
#include <vector>

#include <fst/log.h>
#include <fst/bi-table.h>

namespace fst {

// Stores a collection of non-empty, ordered (multi-)sets with elements of type
// T. A default constructor, operator==, and an STL-style hash functor must be
// defined on the elements. Provides signed integer ID (of type I) for each
// unique set. The IDs are allocated starting from 0 in order.
template <class I, class T>
class Collection {
 public:
  struct Node {  // Trie node.
    I node_id;   // Root is kNoNodeId;
    T element;

    Node() : node_id(kNoNodeId), element(T()) {}

    Node(I i, const T &t) : node_id(i), element(t) {}

    bool operator==(const Node &n) const {
      return n.node_id == node_id && n.element == element;
    }
  };

  struct NodeHash {
    size_t operator()(const Node &n) const {
      static constexpr auto kPrime = 7853;
      return n.node_id + hash_(n.element) * kPrime;
    }
  };

  using NodeTable = CompactHashBiTable<I, Node, NodeHash>;

  class SetIterator {
   public:
    SetIterator(I id, Node node, NodeTable *node_table)
        : id_(id), node_(node), node_table_(node_table) {}

    bool Done() const { return id_ == kNoNodeId; }

    const T &Element() const { return node_.element; }

    void Next() {
      id_ = node_.node_id;
      if (id_ != kNoNodeId) node_ = node_table_->FindEntry(id_);
    }

   private:
    I id_;       // Iterator set node ID.
    Node node_;  // Iterator set node.
    NodeTable *node_table_;
  };

  Collection() {}

  // Looks up integer ID from ordered multi-se, and if it doesn't exist and
  // insert is true, then adds it. Otherwise returns -1.
  I FindId(const std::vector<T> &set, bool insert = true) {
    I node_id = kNoNodeId;
    for (std::ptrdiff_t i = set.size() - 1; i >= 0; --i) {
      Node node(node_id, set[i]);
      node_id = node_table_.FindId(node, insert);
      if (node_id == -1) break;
    }
    return node_id;
  }

  // Finds ordered (multi-)set given integer ID. Returns set iterator to
  // traverse result.
  SetIterator FindSet(I id) {
    if (id < 0 || id >= node_table_.Size()) {
      return SetIterator(kNoNodeId, Node(kNoNodeId, T()), &node_table_);
    } else {
      return SetIterator(id, node_table_.FindEntry(id), &node_table_);
    }
  }

  I Size() const { return node_table_.Size(); }

 private:
  static constexpr I kNoNodeId = -1;
  static const std::hash<T> hash_;

  NodeTable node_table_;
};

template <class I, class T>
constexpr I Collection<I, T>::kNoNodeId;

template <class I, class T>
const std::hash<T> Collection<I, T>::hash_ = {};

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_COLLECTION_H_
