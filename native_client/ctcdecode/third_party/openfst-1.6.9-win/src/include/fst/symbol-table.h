// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes to provide symbol-to-integer and integer-to-symbol mappings.

#ifndef FST_SYMBOL_TABLE_H_
#define FST_SYMBOL_TABLE_H_

#include <functional>
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fst/compat.h>
#include <fst/flags.h>
#include <fst/log.h>
#include <fstream>
#include <map>

DECLARE_bool(fst_compat_symbols);

namespace fst {

constexpr int64_t kNoSymbol = -1;

// WARNING: Reading via symbol table read options should
//          not be used. This is a temporary work around for
//          reading symbol ranges of previously stored symbol sets.
struct SymbolTableReadOptions {
  SymbolTableReadOptions() {}

  SymbolTableReadOptions(
      std::vector<std::pair<int64_t, int64_t>> string_hash_ranges,
      const string &source)
      : string_hash_ranges(std::move(string_hash_ranges)), source(source) {}

  std::vector<std::pair<int64_t, int64_t>> string_hash_ranges;
  string source;
};

struct SymbolTableTextOptions {
  explicit SymbolTableTextOptions(bool allow_negative_labels = false);

  bool allow_negative_labels;
  string fst_field_separator;
};

namespace internal {

// List of symbols with a dense hash for looking up symbol index, rehashing at
// 75% occupancy.
class DenseSymbolMap {
 public:
  DenseSymbolMap();

  DenseSymbolMap(const DenseSymbolMap &x);

  std::pair<int64_t, bool> InsertOrFind(const string &key);

  int64_t Find(const string &key) const;

  size_t Size() const { return symbols_.size(); }

  const string &GetSymbol(size_t idx) const { return symbols_[idx]; }

  void RemoveSymbol(size_t idx);

 private:
  // num_buckets must be power of 2.
  void Rehash(size_t num_buckets);

  int64_t empty_;
  std::vector<string> symbols_;
  std::hash<string> str_hash_;
  std::vector<int64_t> buckets_;
  uint64_t hash_mask_;
};

class SymbolTableImpl {
 public:
  explicit SymbolTableImpl(const string &name)
      : name_(name),
        available_key_(0),
        dense_key_limit_(0),
        check_sum_finalized_(false) {}

  SymbolTableImpl(const SymbolTableImpl &impl)
      : name_(impl.name_),
        available_key_(impl.available_key_),
        dense_key_limit_(impl.dense_key_limit_),
        symbols_(impl.symbols_),
        idx_key_(impl.idx_key_),
        key_map_(impl.key_map_),
        check_sum_finalized_(false) {}

  int64_t AddSymbol(const string &symbol, int64_t key);

  int64_t AddSymbol(const string &symbol) {
    return AddSymbol(symbol, available_key_);
  }

  // Removes the symbol with the given key. The removal is costly
  // (O(NumSymbols)) and may reduce the efficiency of Find() because of a
  // potentially reduced size of the dense key interval.
  void RemoveSymbol(int64_t key);

  static SymbolTableImpl *ReadText(
      std::istream &strm, const string &name,
      const SymbolTableTextOptions &opts = SymbolTableTextOptions());

  static SymbolTableImpl* Read(std::istream &strm,
                               const SymbolTableReadOptions &opts);

  bool Write(std::ostream &strm) const;

  // Return the string associated with the key. If the key is out of
  // range (<0, >max), return an empty string.
  string Find(int64_t key) const {
    int64_t idx = key;
    if (key < 0 || key >= dense_key_limit_) {
      const auto it = key_map_.find(key);
      if (it == key_map_.end()) return "";
      idx = it->second;
    }
    if (idx < 0 || idx >= symbols_.Size()) return "";
    return symbols_.GetSymbol(idx);
  }

  // Returns the key associated with the symbol; if the symbol
  // does not exists, returns kNoSymbol.
  int64_t Find(const string &symbol) const {
    int64_t idx = symbols_.Find(symbol);
    if (idx == kNoSymbol || idx < dense_key_limit_) return idx;
    return idx_key_[idx - dense_key_limit_];
  }

  bool Member(int64_t key) const { return !Find(key).empty(); }

  bool Member(const string &symbol) const { return Find(symbol) != kNoSymbol; }

  int64_t GetNthKey(std::ptrdiff_t pos) const {
    if (pos < 0 || pos >= symbols_.Size()) return kNoSymbol;
    if (pos < dense_key_limit_) return pos;
    return Find(symbols_.GetSymbol(pos));
  }

  const string &Name() const { return name_; }

  void SetName(const string &new_name) { name_ = new_name; }

  const string &CheckSum() const {
    MaybeRecomputeCheckSum();
    return check_sum_string_;
  }

  const string &LabeledCheckSum() const {
    MaybeRecomputeCheckSum();
    return labeled_check_sum_string_;
  }

  int64_t AvailableKey() const { return available_key_; }

  size_t NumSymbols() const { return symbols_.Size(); }

 private:
  // Recomputes the checksums (both of them) if we've had changes since the last
  // computation (i.e., if check_sum_finalized_ is false).
  // Takes ~2.5 microseconds (dbg) or ~230 nanoseconds (opt) on a 2.67GHz Xeon
  // if the checksum is up-to-date (requiring no recomputation).
  void MaybeRecomputeCheckSum() const;

  string name_;
  int64_t available_key_;
  int64_t dense_key_limit_;

  DenseSymbolMap symbols_;
  // Maps index to key for index >= dense_key_limit:
  //   key = idx_key_[index - dense_key_limit]
  std::vector<int64_t> idx_key_;
  // Maps key to index for key >= dense_key_limit_.
  //  index = key_map_[key]
  map<int64_t, int64_t> key_map_;

  mutable bool check_sum_finalized_;
  mutable string check_sum_string_;
  mutable string labeled_check_sum_string_;
  mutable Mutex check_sum_mutex_;
};

}  // namespace internal

// Symbol (string) to integer (and reverse) mapping.
//
// The SymbolTable implements the mappings of labels to strings and reverse.
// SymbolTables are used to describe the alphabet of the input and output
// labels for arcs in a Finite State Transducer.
//
// SymbolTables are reference-counted and can therefore be shared across
// multiple machines. For example a language model grammar G, with a
// SymbolTable for the words in the language model can share this symbol
// table with the lexical representation L o G.
class SymbolTable {
 public:
  // Constructs symbol table with an optional name.
  explicit SymbolTable(const string &name = "<unspecified>")
      : impl_(std::make_shared<internal::SymbolTableImpl>(name)) {}

  virtual ~SymbolTable() {}

  // Reads a text representation of the symbol table from an istream. Pass a
  // name to give the resulting SymbolTable.
  static SymbolTable *ReadText(
      std::istream &strm, const string &name,
      const SymbolTableTextOptions &opts = SymbolTableTextOptions()) {
    auto *impl = internal::SymbolTableImpl::ReadText(strm, name, opts);
    return impl ? new SymbolTable(impl) : nullptr;
  }

  // Reads a text representation of the symbol table.
  static SymbolTable *ReadText(const string &filename,
      const SymbolTableTextOptions &opts = SymbolTableTextOptions()) {
    std::ifstream strm(filename, std::ios_base::in);
    if (!strm.good()) {
      LOG(ERROR) << "SymbolTable::ReadText: Can't open file " << filename;
      return nullptr;
    }
    return ReadText(strm, filename, opts);
  }

  // WARNING: Reading via symbol table read options should not be used. This is
  // a temporary work-around.
  static SymbolTable* Read(std::istream &strm,
                           const SymbolTableReadOptions &opts) {
    auto *impl = internal::SymbolTableImpl::Read(strm, opts);
    return (impl) ? new SymbolTable(impl) : nullptr;
  }

  // Reads a binary dump of the symbol table from a stream.
  static SymbolTable *Read(std::istream &strm,
                           const string &source) {
    SymbolTableReadOptions opts;
    opts.source = source;
    return Read(strm, opts);
  }

  // Reads a binary dump of the symbol table.
  static SymbolTable *Read(const string& filename) {
    std::ifstream strm(filename,
                            std::ios_base::in | std::ios_base::binary);
    if (!strm.good()) {
      LOG(ERROR) << "SymbolTable::Read: Can't open file " << filename;
      return nullptr;
    }
    return Read(strm, filename);
  }

  // DERIVABLE INTERFACE

  // Creates a reference counted copy.
  virtual SymbolTable *Copy() const { return new SymbolTable(*this); }

  // Adds a symbol with given key to table. A symbol table also keeps track of
  // the last available key (highest key value in the symbol table).
  virtual int64_t AddSymbol(const string &symbol, int64_t key) {
    MutateCheck();
    return impl_->AddSymbol(symbol, key);
  }

  // Adds a symbol to the table. The associated value key is automatically
  // assigned by the symbol table.
  virtual int64_t AddSymbol(const string &symbol) {
    MutateCheck();
    return impl_->AddSymbol(symbol);
  }

  // Adds another symbol table to this table. All key values will be offset
  // by the current available key (highest key value in the symbol table).
  // Note string symbols with the same key value will still have the same
  // key value after the symbol table has been merged, but a different
  // value. Adding symbol tables do not result in changes in the base table.
  virtual void AddTable(const SymbolTable &table);

  // Returns the current available key (i.e., highest key + 1) in the symbol
  // table.
  virtual int64_t AvailableKey() const { return impl_->AvailableKey(); }

  // Return the label-agnostic MD5 check-sum for this table. All new symbols
  // added to the table will result in an updated checksum. Deprecated.
  virtual const string &CheckSum() const { return impl_->CheckSum(); }

  virtual int64_t GetNthKey(std::ptrdiff_t pos) const { return impl_->GetNthKey(pos); }

  // Returns the string associated with the key; if the key is out of
  // range (<0, >max), returns an empty string.
  virtual string Find(int64_t key) const { return impl_->Find(key); }

  // Returns the key associated with the symbol; if the symbol does not exist,
  // kNoSymbol is returned.
  virtual int64_t Find(const string &symbol) const { return impl_->Find(symbol); }

  // Same as CheckSum(), but returns an label-dependent version.
  virtual const string &LabeledCheckSum() const {
    return impl_->LabeledCheckSum();
  }

  virtual bool Member(int64_t key) const { return impl_->Member(key); }

  virtual bool Member(const string &symbol) const {
    return impl_->Member(symbol);
  }

  // Returns the name of the symbol table.
  virtual const string &Name() const { return impl_->Name(); }

  // Returns the current number of symbols in table (not necessarily equal to
  // AvailableKey()).
  virtual size_t NumSymbols() const { return impl_->NumSymbols(); }

  virtual void RemoveSymbol(int64_t key) {
    MutateCheck();
    return impl_->RemoveSymbol(key);
  }

  // Sets the name of the symbol table.
  virtual void SetName(const string &new_name) {
    MutateCheck();
    impl_->SetName(new_name);
  }

  virtual bool Write(std::ostream &strm) const { return impl_->Write(strm); }

  virtual bool Write(const string &filename) const {
    std::ofstream strm(filename,
                             std::ios_base::out | std::ios_base::binary);
    if (!strm.good()) {
      LOG(ERROR) << "SymbolTable::Write: Can't open file " << filename;
      return false;
    }
    return Write(strm);
  }

  // Dump a text representation of the symbol table via a stream.
  virtual bool WriteText(std::ostream &strm,
      const SymbolTableTextOptions &opts = SymbolTableTextOptions()) const;

  // Dump an text representation of the symbol table.
  virtual bool WriteText(const string &filename) const {
    std::ofstream strm(filename);
    if (!strm.good()) {
      LOG(ERROR) << "SymbolTable::WriteText: Can't open file " << filename;
      return false;
    }
    return WriteText(strm);
  }

 private:
  explicit SymbolTable(internal::SymbolTableImpl *impl) : impl_(impl) {}

  void MutateCheck() {
    if (!impl_.unique()) impl_.reset(new internal::SymbolTableImpl(*impl_));
  }

  const internal::SymbolTableImpl *Impl() const { return impl_.get(); }

 private:
  std::shared_ptr<internal::SymbolTableImpl> impl_;
};

// Iterator class for symbols in a symbol table.
class SymbolTableIterator {
 public:
  explicit SymbolTableIterator(const SymbolTable &table)
      : table_(table),
        pos_(0),
        nsymbols_(table.NumSymbols()),
        key_(table.GetNthKey(0)) {}

  ~SymbolTableIterator() {}

  // Returns whether iterator is done.
  bool Done() const { return (pos_ == nsymbols_); }

  // Return the key of the current symbol.
  int64_t Value() const { return key_; }

  // Return the string of the current symbol.
  string Symbol() const { return table_.Find(key_); }

  // Advances iterator.
  void Next() {
    ++pos_;
    if (pos_ < nsymbols_) key_ = table_.GetNthKey(pos_);
  }

  // Resets iterator.
  void Reset() {
    pos_ = 0;
    key_ = table_.GetNthKey(0);
  }

 private:
  const SymbolTable &table_;
  std::ptrdiff_t pos_;
  size_t nsymbols_;
  int64_t key_;
};

// Relabels a symbol table as specified by the input vector of pairs
// (old label, new label). The new symbol table only retains symbols
// for which a relabeling is explicitly specified.
//
// TODO(allauzen): consider adding options to allow for some form of implicit
// identity relabeling.
template <class Label>
SymbolTable *RelabelSymbolTable(const SymbolTable *table,
    const std::vector<std::pair<Label, Label>> &pairs) {
  auto *new_table = new SymbolTable(
      table->Name().empty() ? string()
                            : (string("relabeled_") + table->Name()));
  for (const auto &pair : pairs) {
    new_table->AddSymbol(table->Find(pair.first), pair.second);
  }
  return new_table;
}

// Returns true if the two symbol tables have equal checksums. Passing in
// nullptr for either table always returns true.
bool CompatSymbols(const SymbolTable *syms1, const SymbolTable *syms2,
                   bool warning = true);

// Symbol Table serialization.

void SymbolTableToString(const SymbolTable *table, string *result);

SymbolTable *StringToSymbolTable(const string &str);

}  // namespace fst

#endif  // FST_SYMBOL_TABLE_H_
