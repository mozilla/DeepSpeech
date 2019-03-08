// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes for representing a bijective mapping between an arbitrary entry
// of type T and a signed integral ID.

#ifndef FST_BI_TABLE_H_
#define FST_BI_TABLE_H_

#include <deque>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fst/log.h>
#include <fst/memory.h>

namespace fst {

// Bitables model bijective mappings between entries of an arbitrary type T and
// an signed integral ID of type I. The IDs are allocated starting from 0 in
// order.
//
// template <class I, class T>
// class BiTable {
//  public:
//
//   // Required constructors.
//   BiTable();
//
//   // Looks up integer ID from entry. If it doesn't exist and insert
//   / is true, adds it; otherwise, returns -1.
//   I FindId(const T &entry, bool insert = true);
//
//   // Looks up entry from integer ID.
//   const T &FindEntry(I) const;
//
//   // Returns number of stored entries.
//   I Size() const;
// };

// An implementation using a hash map for the entry to ID mapping. H is the
// hash function and E is the equality function. If passed to the constructor,
// ownership is given to this class.
template <class I, class T, class H, class E = std::equal_to<T>>
class HashBiTable {
 public:
  // Reserves space for table_size elements. If passing H and E to the
  // constructor, this class owns them.
  explicit HashBiTable(size_t table_size = 0, H *h = nullptr, E *e = nullptr) :
      hash_func_(h ? h : new H()), hash_equal_(e ? e : new E()),
      entry2id_(table_size, *hash_func_, *hash_equal_) {
    if (table_size) id2entry_.reserve(table_size);
  }

  HashBiTable(const HashBiTable<I, T, H, E> &table)
      : hash_func_(new H(*table.hash_func_)),
        hash_equal_(new E(*table.hash_equal_)),
        entry2id_(table.entry2id_.begin(), table.entry2id_.end(),
                  table.entry2id_.size(), *hash_func_, *hash_equal_),
        id2entry_(table.id2entry_) {}

  I FindId(const T &entry, bool insert = true) {
    if (!insert) {
      const auto it = entry2id_.find(entry);
      return it == entry2id_.end() ? -1 : it->second - 1;
    }
    I &id_ref = entry2id_[entry];
    if (id_ref == 0) {  // T not found; stores and assigns a new ID.
      id2entry_.push_back(entry);
      id_ref = id2entry_.size();
    }
    return id_ref - 1;  // NB: id_ref = ID + 1.
  }

  const T &FindEntry(I s) const { return id2entry_[s]; }

  I Size() const { return id2entry_.size(); }

  // TODO(riley): Add fancy clear-to-size, as in CompactHashBiTable.
  void Clear() {
    entry2id_.clear();
    id2entry_.clear();
  }

 private:
  std::unique_ptr<H> hash_func_;
  std::unique_ptr<E> hash_equal_;
  std::unordered_map<T, I, H, E> entry2id_;
  std::vector<T> id2entry_;
};

// Enables alternative hash set representations below.
enum HSType { HS_STL = 0, HS_DENSE = 1, HS_SPARSE = 2, HS_FLAT = 3 };

// Default hash set is STL hash_set.
template <class K, class H, class E, HSType HS>
struct HashSet : public std::unordered_set<K, H, E, PoolAllocator<K>> {
  explicit HashSet(size_t n = 0, const H &h = H(), const E &e = E())
      : std::unordered_set<K, H, E, PoolAllocator<K>>(n, h, e) {}

  void rehash(size_t n) {}
};

// An implementation using a hash set for the entry to ID mapping. The hash set
// holds keys which are either the ID or kCurrentKey. These keys can be mapped
// to entries either by looking up in the entry vector or, if kCurrentKey, in
// current_entry_. The hash and key equality functions map to entries first. H
// is the hash function and E is the equality function. If passed to the
// constructor, ownership is given to this class.
template <class I, class T, class H, class E = std::equal_to<T>,
          HSType HS = HS_FLAT>
class CompactHashBiTable {
 public:
  friend class HashFunc;
  friend class HashEqual;

  // Reserves space for table_size elements. If passing H and E to the
  // constructor, this class owns them.
  explicit CompactHashBiTable(size_t table_size = 0, H *h = nullptr,
                              E *e = nullptr) :
        hash_func_(h ? h : new H()), hash_equal_(e ? e : new E()),
        compact_hash_func_(*this), compact_hash_equal_(*this),
        keys_(table_size, compact_hash_func_, compact_hash_equal_) {
    if (table_size) id2entry_.reserve(table_size);
  }

  CompactHashBiTable(const CompactHashBiTable<I, T, H, E, HS> &table)
      : hash_func_(new H(*table.hash_func_)),
        hash_equal_(new E(*table.hash_equal_)),
        compact_hash_func_(*this), compact_hash_equal_(*this),
        keys_(table.keys_.size(), compact_hash_func_, compact_hash_equal_),
        id2entry_(table.id2entry_) {
    keys_.insert(table.keys_.begin(), table.keys_.end());
  }

  I FindId(const T &entry, bool insert = true) {
    current_entry_ = &entry;
    if (insert) {
      auto result = keys_.insert(kCurrentKey);
      if (!result.second) return *result.first;  // Already exists.
      // Overwrites kCurrentKey with a new key value; this is safe because it
      // doesn't affect hashing or equality testing.
      I key = id2entry_.size();
      const_cast<I &>(*result.first) = key;
      id2entry_.push_back(entry);
      return key;
    }
    const auto it = keys_.find(kCurrentKey);
    return it == keys_.end() ? -1 : *it;
  }

  const T &FindEntry(I s) const { return id2entry_[s]; }

  I Size() const { return id2entry_.size(); }

  // Clears content; with argument, erases last n IDs.
  void Clear(std::ptrdiff_t n = -1) {
    if (n < 0 || n >= id2entry_.size()) {  // Clears completely.
      keys_.clear();
      id2entry_.clear();
    } else if (n == id2entry_.size() - 1) {  // Leaves only key 0.
      const T entry = FindEntry(0);
      keys_.clear();
      id2entry_.clear();
      FindId(entry, true);
    } else {
      while (n-- > 0) {
        I key = id2entry_.size() - 1;
        keys_.erase(key);
        id2entry_.pop_back();
      }
      keys_.rehash(0);
    }
  }

 private:
  static constexpr I kCurrentKey = -1;
  static constexpr I kEmptyKey = -2;
  static constexpr I kDeletedKey = -3;

  class HashFunc {
   public:
    explicit HashFunc(const CompactHashBiTable &ht) : ht_(&ht) {}

    size_t operator()(I k) const {
      if (k >= kCurrentKey) {
        return (*ht_->hash_func_)(ht_->Key2Entry(k));
      } else {
        return 0;
      }
    }

   private:
    const CompactHashBiTable *ht_;
  };

  class HashEqual {
   public:
    explicit HashEqual(const CompactHashBiTable &ht) : ht_(&ht) {}

    bool operator()(I k1, I k2) const {
      if (k1 == k2) {
        return true;
      } else if (k1 >= kCurrentKey && k2 >= kCurrentKey) {
        return (*ht_->hash_equal_)(ht_->Key2Entry(k1), ht_->Key2Entry(k2));
      } else {
        return false;
      }
    }

   private:
    const CompactHashBiTable *ht_;
  };

  using KeyHashSet = HashSet<I, HashFunc, HashEqual, HS>;

  const T &Key2Entry(I k) const {
    if (k == kCurrentKey) {
      return *current_entry_;
    } else {
      return id2entry_[k];
    }
  }

  std::unique_ptr<H> hash_func_;
  std::unique_ptr<E> hash_equal_;
  HashFunc compact_hash_func_;
  HashEqual compact_hash_equal_;
  KeyHashSet keys_;
  std::vector<T> id2entry_;
  const T *current_entry_;
};

template <class I, class T, class H, class E, HSType HS>
constexpr I CompactHashBiTable<I, T, H, E, HS>::kCurrentKey;

template <class I, class T, class H, class E, HSType HS>
constexpr I CompactHashBiTable<I, T, H, E, HS>::kEmptyKey;

template <class I, class T, class H, class E, HSType HS>
constexpr I CompactHashBiTable<I, T, H, E, HS>::kDeletedKey;

// An implementation using a vector for the entry to ID mapping. It is passed a
// function object FP that should fingerprint entries uniquely to an integer
// that can used as a vector index. Normally, VectorBiTable constructs the FP
// object. The user can instead pass in this object; in that case, VectorBiTable
// takes its ownership.
template <class I, class T, class FP>
class VectorBiTable {
 public:
  // Reserves table_size cells of space. If passing FP argument to the
  // constructor, this class owns it.
  explicit VectorBiTable(FP *fp = nullptr, size_t table_size = 0) :
      fp_(fp ? fp : new FP()) {
    if (table_size) id2entry_.reserve(table_size);
  }

  VectorBiTable(const VectorBiTable<I, T, FP> &table)
      : fp_(new FP(*table.fp_)), fp2id_(table.fp2id_),
        id2entry_(table.id2entry_) {}

  I FindId(const T &entry, bool insert = true) {
    std::ptrdiff_t fp = (*fp_)(entry);
    if (fp >= fp2id_.size()) fp2id_.resize(fp + 1);
    I &id_ref = fp2id_[fp];
    if (id_ref == 0) {  // T not found.
      if (insert) {     // Stores and assigns a new ID.
        id2entry_.push_back(entry);
        id_ref = id2entry_.size();
      } else {
        return -1;
      }
    }
    return id_ref - 1;  // NB: id_ref = ID + 1.
  }

  const T &FindEntry(I s) const { return id2entry_[s]; }

  I Size() const { return id2entry_.size(); }

  const FP &Fingerprint() const { return *fp_; }

 private:
  std::unique_ptr<FP> fp_;
  std::vector<I> fp2id_;
  std::vector<T> id2entry_;
};

// An implementation using a vector and a compact hash table. The selecting
// functor S returns true for entries to be hashed in the vector. The
// fingerprinting functor FP returns a unique fingerprint for each entry to be
// hashed in the vector (these need to be suitable for indexing in a vector).
// The hash functor H is used when hashing entry into the compact hash table.
// If passed to the constructor, ownership is given to this class.
template <class I, class T, class S, class FP, class H, HSType HS = HS_DENSE>
class VectorHashBiTable {
 public:
  friend class HashFunc;
  friend class HashEqual;

  explicit VectorHashBiTable(S *s, FP *fp, H *h, size_t vector_size = 0,
                             size_t entry_size = 0)
      : selector_(s), fp_(fp), h_(h), hash_func_(*this), hash_equal_(*this),
        keys_(0, hash_func_, hash_equal_) {
    if (vector_size) fp2id_.reserve(vector_size);
    if (entry_size) id2entry_.reserve(entry_size);
  }

  VectorHashBiTable(const VectorHashBiTable<I, T, S, FP, H, HS> &table)
      : selector_(new S(table.s_)), fp_(new FP(*table.fp_)),
        h_(new H(*table.h_)), id2entry_(table.id2entry_),
        fp2id_(table.fp2id_), hash_func_(*this), hash_equal_(*this),
        keys_(table.keys_.size(), hash_func_, hash_equal_) {
    keys_.insert(table.keys_.begin(), table.keys_.end());
  }

  I FindId(const T &entry, bool insert = true) {
    if ((*selector_)(entry)) {  // Uses the vector if selector_(entry) == true.
      uint64_t fp = (*fp_)(entry);
      if (fp2id_.size() <= fp) fp2id_.resize(fp + 1, 0);
      if (fp2id_[fp] == 0) {  // T not found.
        if (insert) {         // Stores and assigns a new ID.
          id2entry_.push_back(entry);
          fp2id_[fp] = id2entry_.size();
        } else {
          return -1;
        }
      }
      return fp2id_[fp] - 1;  // NB: assoc_value = ID + 1.
    } else {                  // Uses the hash table otherwise.
      current_entry_ = &entry;
      const auto it = keys_.find(kCurrentKey);
      if (it == keys_.end()) {
        if (insert) {
          I key = id2entry_.size();
          id2entry_.push_back(entry);
          keys_.insert(key);
          return key;
        } else {
          return -1;
        }
      } else {
        return *it;
      }
    }
  }

  const T &FindEntry(I s) const { return id2entry_[s]; }

  I Size() const { return id2entry_.size(); }

  const S &Selector() const { return *selector_; }

  const FP &Fingerprint() const { return *fp_; }

  const H &Hash() const { return *h_; }

 private:
  static constexpr I kCurrentKey = -1;
  static constexpr I kEmptyKey = -2;

  class HashFunc {
   public:
    explicit HashFunc(const VectorHashBiTable &ht) : ht_(&ht) {}

    size_t operator()(I k) const {
      if (k >= kCurrentKey) {
        return (*(ht_->h_))(ht_->Key2Entry(k));
      } else {
        return 0;
      }
    }

   private:
    const VectorHashBiTable *ht_;
  };

  class HashEqual {
   public:
    explicit HashEqual(const VectorHashBiTable &ht) : ht_(&ht) {}

    bool operator()(I k1, I k2) const {
      if (k1 >= kCurrentKey && k2 >= kCurrentKey) {
        return ht_->Key2Entry(k1) == ht_->Key2Entry(k2);
      } else {
        return k1 == k2;
      }
    }

   private:
    const VectorHashBiTable *ht_;
  };

  using KeyHashSet = HashSet<I, HashFunc, HashEqual, HS>;

  const T &Key2Entry(I k) const {
    if (k == kCurrentKey) {
      return *current_entry_;
    } else {
      return id2entry_[k];
    }
  }

  std::unique_ptr<S> selector_;  // True if entry hashed into vector.
  std::unique_ptr<FP> fp_;       // Fingerprint used for hashing into vector.
  std::unique_ptr<H> h_;         // Hash funcion used for hashing into hash_set.

  std::vector<T> id2entry_;  // Maps state IDs to entry.
  std::vector<I> fp2id_;     // Maps entry fingerprints to IDs.

  // Compact implementation of the hash table mapping entries to state IDs
  // using the hash function h_.
  HashFunc hash_func_;
  HashEqual hash_equal_;
  KeyHashSet keys_;
  const T *current_entry_;
};

template <class I, class T, class S, class FP, class H, HSType HS>
constexpr I VectorHashBiTable<I, T, S, FP, H, HS>::kCurrentKey;

template <class I, class T, class S, class FP, class H, HSType HS>
constexpr I VectorHashBiTable<I, T, S, FP, H, HS>::kEmptyKey;

// An implementation using a hash map for the entry to ID mapping. This version
// permits erasing of arbitrary states. The entry T must have == defined and
// its default constructor must produce a entry that will never be seen. F is
// the hash function.
template <class I, class T, class F>
class ErasableBiTable {
 public:
  ErasableBiTable() : first_(0) {}

  I FindId(const T &entry, bool insert = true) {
    I &id_ref = entry2id_[entry];
    if (id_ref == 0) {  // T not found.
      if (insert) {     // Stores and assigns a new ID.
        id2entry_.push_back(entry);
        id_ref = id2entry_.size() + first_;
      } else {
        return -1;
      }
    }
    return id_ref - 1;  // NB: id_ref = ID + 1.
  }

  const T &FindEntry(I s) const { return id2entry_[s - first_]; }

  I Size() const { return id2entry_.size(); }

  void Erase(I s) {
    auto &ref = id2entry_[s - first_];
    entry2id_.erase(ref);
    ref = empty_entry_;
    while (!id2entry_.empty() && id2entry_.front() == empty_entry_) {
      id2entry_.pop_front();
      ++first_;
    }
  }

 private:
  std::unordered_map<T, I, F> entry2id_;
  std::deque<T> id2entry_;
  const T empty_entry_;
  I first_;  // I of first element in the deque.
};

}  // namespace fst

#endif  // FST_BI_TABLE_H_
