// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes to provide symbol-to-integer and integer-to-symbol mappings.

#include <fst/symbol-table.h>

#include <fst/flags.h>
#include <fst/log.h>

#include <fstream>
#include <fst/util.h>

DEFINE_bool(fst_compat_symbols, true,
            "Require symbol tables to match when appropriate");
DEFINE_string(fst_field_separator, "\t ",
              "Set of characters used as a separator between printed fields");

namespace fst {

SymbolTableTextOptions::SymbolTableTextOptions(bool allow_negative_labels)
    : allow_negative_labels(allow_negative_labels),
      fst_field_separator(FLAGS_fst_field_separator) {}

namespace internal {

// Maximum line length in textual symbols file.
static constexpr int kLineLen = 8096;

// Identifies stream data as a symbol table (and its endianity).
static constexpr int32_t kSymbolTableMagicNumber = 2125658996;

DenseSymbolMap::DenseSymbolMap()
    : empty_(-1), buckets_(1 << 4), hash_mask_(buckets_.size() - 1) {
  std::uninitialized_fill(buckets_.begin(), buckets_.end(), empty_);
}

DenseSymbolMap::DenseSymbolMap(const DenseSymbolMap &other)
    : empty_(-1),
      symbols_(other.symbols_),
      buckets_(other.buckets_),
      hash_mask_(other.hash_mask_) {}

std::pair<int64_t, bool> DenseSymbolMap::InsertOrFind(const string &key) {
  static constexpr float kMaxOccupancyRatio = 0.75;  // Grows when 75% full.
  if (Size() >= kMaxOccupancyRatio * buckets_.size()) {
    Rehash(buckets_.size() * 2);
  }
  size_t idx = str_hash_(key) & hash_mask_;
  while (buckets_[idx] != empty_) {
    const auto stored_value = buckets_[idx];
    if (symbols_[stored_value] == key) return {stored_value, false};
    idx = (idx + 1) & hash_mask_;
  }
  const auto next = Size();
  buckets_[idx] = next;
  symbols_.emplace_back(key);
  return {next, true};
}

int64_t DenseSymbolMap::Find(const string &key) const {
  size_t idx = str_hash_(key) & hash_mask_;
  while (buckets_[idx] != empty_) {
    const auto stored_value = buckets_[idx];
    if (symbols_[stored_value] == key) return stored_value;
    idx = (idx + 1) & hash_mask_;
  }
  return buckets_[idx];
}

void DenseSymbolMap::Rehash(size_t num_buckets) {
  buckets_.resize(num_buckets);
  hash_mask_ = buckets_.size() - 1;
  std::uninitialized_fill(buckets_.begin(), buckets_.end(), empty_);
  for (size_t i = 0; i < Size(); ++i) {
    size_t idx = str_hash_(string(symbols_[i])) & hash_mask_;
    while (buckets_[idx] != empty_) {
      idx = (idx + 1) & hash_mask_;
    }
    buckets_[idx] = i;
  }
}

void DenseSymbolMap::RemoveSymbol(size_t idx) {
  symbols_.erase(symbols_.begin() + idx);
  Rehash(buckets_.size());
}

SymbolTableImpl *SymbolTableImpl::ReadText(std::istream &strm,
                                           const string &filename,
                                           const SymbolTableTextOptions &opts) {
  std::unique_ptr<SymbolTableImpl> impl(new SymbolTableImpl(filename));
  int64_t nline = 0;
  char line[kLineLen];
  while (!strm.getline(line, kLineLen).fail()) {
    ++nline;
    std::vector<char *> col;
    const auto separator = opts.fst_field_separator + "\n";
    SplitString(line, separator.c_str(), &col, true);
    if (col.empty()) continue;  // Empty line.
    if (col.size() != 2) {
      LOG(ERROR) << "SymbolTable::ReadText: Bad number of columns ("
                 << col.size() << "), "
                 << "file = " << filename << ", line = " << nline << ":<"
                 << line << ">";
      return nullptr;
    }
    const char *symbol = col[0];
    const char *value = col[1];
    char *p;
    const auto key = strtoll(value, &p, 10);
    if (p < value + strlen(value) || (!opts.allow_negative_labels && key < 0) ||
        key == kNoSymbol) {
      LOG(ERROR) << "SymbolTable::ReadText: Bad non-negative integer \""
                 << value << "\", "
                 << "file = " << filename << ", line = " << nline;
      return nullptr;
    }
    impl->AddSymbol(symbol, key);
  }
  return impl.release();
}

void SymbolTableImpl::MaybeRecomputeCheckSum() const {
  {
    ReaderMutexLock check_sum_lock(&check_sum_mutex_);
    if (check_sum_finalized_) return;
  }
  // We'll acquire an exclusive lock to recompute the checksums.
  MutexLock check_sum_lock(&check_sum_mutex_);
  if (check_sum_finalized_) {  // Another thread (coming in around the same time
    return;                    // might have done it already). So we recheck.
  }
  // Calculates the original label-agnostic checksum.
  CheckSummer check_sum;
  for (size_t i = 0; i < symbols_.Size(); ++i) {
    const auto &symbol = symbols_.GetSymbol(i);
    check_sum.Update(symbol.data(), symbol.size());
    check_sum.Update("", 1);
  }
  check_sum_string_ = check_sum.Digest();
  // Calculates the safer, label-dependent checksum.
  CheckSummer labeled_check_sum;
  for (int64_t i = 0; i < dense_key_limit_; ++i) {
    std::ostringstream line;
    line << symbols_.GetSymbol(i) << '\t' << i;
    labeled_check_sum.Update(line.str().data(), line.str().size());
  }
  using citer = map<int64_t, int64_t>::const_iterator;
  for (citer it = key_map_.begin(); it != key_map_.end(); ++it) {
    // TODO(tombagby, 2013-11-22) This line maintains a bug that ignores
    // negative labels in the checksum that too many tests rely on.
    if (it->first < dense_key_limit_) continue;
    std::ostringstream line;
    line << symbols_.GetSymbol(it->second) << '\t' << it->first;
    labeled_check_sum.Update(line.str().data(), line.str().size());
  }
  labeled_check_sum_string_ = labeled_check_sum.Digest();
  check_sum_finalized_ = true;
}

int64_t SymbolTableImpl::AddSymbol(const string &symbol, int64_t key) {
  if (key == kNoSymbol) return key;
  const auto insert_key = symbols_.InsertOrFind(symbol);
  if (!insert_key.second) {
    const auto key_already = GetNthKey(insert_key.first);
    if (key_already == key) return key;
    VLOG(1) << "SymbolTable::AddSymbol: symbol = " << symbol
            << " already in symbol_map_ with key = " << key_already
            << " but supplied new key = " << key << " (ignoring new key)";
    return key_already;
  }
  if (key == (symbols_.Size() - 1) && key == dense_key_limit_) {
    ++dense_key_limit_;
  } else {
    idx_key_.push_back(key);
    key_map_[key] = symbols_.Size() - 1;
  }
  if (key >= available_key_) available_key_ = key + 1;
  check_sum_finalized_ = false;
  return key;
}

// TODO(rybach): Consider a more efficient implementation which re-uses holes in
// the dense-key range or re-arranges the dense-key range from time to time.
void SymbolTableImpl::RemoveSymbol(const int64_t key) {
  auto idx = key;
  if (key < 0 || key >= dense_key_limit_) {
    auto iter = key_map_.find(key);
    if (iter == key_map_.end()) return;
    idx = iter->second;
    key_map_.erase(iter);
  }
  if (idx < 0 || idx >= symbols_.Size()) return;
  symbols_.RemoveSymbol(idx);
  // Removed one symbol, all indexes > idx are shifted by -1.
  for (auto &k : key_map_) {
    if (k.second > idx) --k.second;
  }
  if (key >= 0 && key < dense_key_limit_) {
    // Removal puts a hole in the dense key range. Adjusts range to [0, key).
    const auto new_dense_key_limit = key;
    for (int64_t i = key + 1; i < dense_key_limit_; ++i) {
      key_map_[i] = i - 1;
    }
    // Moves existing values in idx_key to new place.
    idx_key_.resize(symbols_.Size() - new_dense_key_limit);
    for (int64_t i = symbols_.Size(); i >= dense_key_limit_; --i) {
      idx_key_[i - new_dense_key_limit - 1] = idx_key_[i - dense_key_limit_];
    }
    // Adds indexes for previously dense keys.
    for (int64_t i = new_dense_key_limit; i < dense_key_limit_ - 1; ++i) {
      idx_key_[i - new_dense_key_limit] = i + 1;
    }
    dense_key_limit_ = new_dense_key_limit;
  } else {
    // Remove entry for removed index in idx_key.
    for (int64_t i = idx - dense_key_limit_; i < idx_key_.size() - 1; ++i) {
      idx_key_[i] = idx_key_[i + 1];
    }
    idx_key_.pop_back();
  }
  if (key == available_key_ - 1) available_key_ = key;
}

SymbolTableImpl *SymbolTableImpl::Read(std::istream &strm,
                                       const SymbolTableReadOptions &opts) {
  int32_t magic_number = 0;
  ReadType(strm, &magic_number);
  if (strm.fail()) {
    LOG(ERROR) << "SymbolTable::Read: Read failed";
    return nullptr;
  }
  string name;
  ReadType(strm, &name);
  std::unique_ptr<SymbolTableImpl> impl(new SymbolTableImpl(name));
  ReadType(strm, &impl->available_key_);
  int64_t size;
  ReadType(strm, &size);
  if (strm.fail()) {
    LOG(ERROR) << "SymbolTable::Read: Read failed";
    return nullptr;
  }
  string symbol;
  int64_t key;
  impl->check_sum_finalized_ = false;
  for (int64_t i = 0; i < size; ++i) {
    ReadType(strm, &symbol);
    ReadType(strm, &key);
    if (strm.fail()) {
      LOG(ERROR) << "SymbolTable::Read: Read failed";
      return nullptr;
    }
    impl->AddSymbol(symbol, key);
  }
  return impl.release();
}

bool SymbolTableImpl::Write(std::ostream &strm) const {
  WriteType(strm, kSymbolTableMagicNumber);
  WriteType(strm, name_);
  WriteType(strm, available_key_);
  const int64_t size = symbols_.Size();
  WriteType(strm, size);
  for (int64_t i = 0; i < size; ++i) {
    auto key = (i < dense_key_limit_) ? i : idx_key_[i - dense_key_limit_];
    WriteType(strm, symbols_.GetSymbol(i));
    WriteType(strm, key);
  }
  strm.flush();
  if (strm.fail()) {
    LOG(ERROR) << "SymbolTable::Write: Write failed";
    return false;
  }
  return true;
}

}  // namespace internal

void SymbolTable::AddTable(const SymbolTable &table) {
  MutateCheck();
  for (SymbolTableIterator iter(table); !iter.Done(); iter.Next()) {
    impl_->AddSymbol(iter.Symbol());
  }
}

bool SymbolTable::WriteText(std::ostream &strm,
                            const SymbolTableTextOptions &opts) const {
  if (opts.fst_field_separator.empty()) {
    LOG(ERROR) << "Missing required field separator";
    return false;
  }
  bool once_only = false;
  for (SymbolTableIterator iter(*this); !iter.Done(); iter.Next()) {
    std::ostringstream line;
    if (iter.Value() < 0 && !opts.allow_negative_labels && !once_only) {
      LOG(WARNING) << "Negative symbol table entry when not allowed";
      once_only = true;
    }
    line << iter.Symbol() << opts.fst_field_separator[0] << iter.Value()
         << '\n';
    strm.write(line.str().data(), line.str().length());
  }
  return true;
}

bool CompatSymbols(const SymbolTable *syms1, const SymbolTable *syms2,
                   bool warning) {
  // Flag can explicitly override this check.
  if (!FLAGS_fst_compat_symbols) return true;
  if (syms1 && syms2 &&
      (syms1->LabeledCheckSum() != syms2->LabeledCheckSum())) {
    if (warning) {
      LOG(WARNING) << "CompatSymbols: Symbol table checksums do not match. "
                   << "Table sizes are " << syms1->NumSymbols() << " and "
                   << syms2->NumSymbols();
    }
    return false;
  } else {
    return true;
  }
}

void SymbolTableToString(const SymbolTable *table, string *result) {
  std::ostringstream ostrm;
  table->Write(ostrm);
  *result = ostrm.str();
}

SymbolTable *StringToSymbolTable(const string &str) {
  std::istringstream istrm(str);
  return SymbolTable::Read(istrm, SymbolTableReadOptions());
}

}  // namespace fst
