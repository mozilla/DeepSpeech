// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//

#include <fst/symbol-table-ops.h>

#include <string>

namespace fst {

SymbolTable *MergeSymbolTable(const SymbolTable &left, const SymbolTable &right,
                              bool *right_relabel_output) {
  // MergeSymbolTable detects several special cases.  It will return a reference
  // copied version of SymbolTable of left or right if either symbol table is
  // a superset of the other.
  std::unique_ptr<SymbolTable> merged(
      new SymbolTable("merge_" + left.Name() + "_" + right.Name()));
  // Copies everything from the left symbol table.
  bool left_has_all = true;
  bool right_has_all = true;
  bool relabel = false;
  for (SymbolTableIterator liter(left); !liter.Done(); liter.Next()) {
    merged->AddSymbol(liter.Symbol(), liter.Value());
    if (right_has_all) {
      int64_t key = right.Find(liter.Symbol());
      if (key == -1) {
        right_has_all = false;
      } else if (!relabel && key != liter.Value()) {
        relabel = true;
      }
    }
  }
  if (right_has_all) {
    if (right_relabel_output) *right_relabel_output = relabel;
    return right.Copy();
  }
  // add all symbols we can from right symbol table
  std::vector<string> conflicts;
  for (SymbolTableIterator riter(right); !riter.Done(); riter.Next()) {
    int64_t key = merged->Find(riter.Symbol());
    if (key != -1) {
      // Symbol already exists, maybe with different value
      if (key != riter.Value()) relabel = true;
      continue;
    }
    // Symbol doesn't exist from left
    left_has_all = false;
    if (!merged->Find(riter.Value()).empty()) {
      // we can't add this where we want to, add it later, in order
      conflicts.push_back(riter.Symbol());
      continue;
    }
    // there is a hole and we can add this symbol with its id
    merged->AddSymbol(riter.Symbol(), riter.Value());
  }
  if (right_relabel_output) *right_relabel_output = relabel;
  if (left_has_all) return left.Copy();
  // Add all symbols that conflicted, in order
  for (const auto &conflict : conflicts) merged->AddSymbol(conflict);
  return merged.release();
}

SymbolTable *CompactSymbolTable(const SymbolTable &syms) {
  std::map<int64_t, string> sorted;
  SymbolTableIterator stiter(syms);
  for (; !stiter.Done(); stiter.Next()) {
    sorted[stiter.Value()] = stiter.Symbol();
  }
  auto *compact = new SymbolTable(syms.Name() + "_compact");
  int64_t newkey = 0;
  for (const auto &kv : sorted) compact->AddSymbol(kv.second, newkey++);
  return compact;
}

SymbolTable *FstReadSymbols(const string &filename, bool input_symbols) {
  std::ifstream in(filename, std::ios_base::in | std::ios_base::binary);
  if (!in) {
    LOG(ERROR) << "FstReadSymbols: Can't open file " << filename;
    return nullptr;
  }
  FstHeader hdr;
  if (!hdr.Read(in, filename)) {
    LOG(ERROR) << "FstReadSymbols: Couldn't read header from " << filename;
    return nullptr;
  }
  if (hdr.GetFlags() & FstHeader::HAS_ISYMBOLS) {
    std::unique_ptr<SymbolTable> isymbols(SymbolTable::Read(in, filename));
    if (isymbols == nullptr) {
      LOG(ERROR) << "FstReadSymbols: Couldn't read input symbols from "
                 << filename;
      return nullptr;
    }
    if (input_symbols) return isymbols.release();
  }
  if (hdr.GetFlags() & FstHeader::HAS_OSYMBOLS) {
    std::unique_ptr<SymbolTable> osymbols(SymbolTable::Read(in, filename));
    if (osymbols == nullptr) {
      LOG(ERROR) << "FstReadSymbols: Couldn't read output symbols from "
                 << filename;
      return nullptr;
    }
    if (!input_symbols) return osymbols.release();
  }
  LOG(ERROR) << "FstReadSymbols: The file " << filename
             << " doesn't contain the requested symbols";
  return nullptr;
}

bool AddAuxiliarySymbols(const string &prefix, int64_t start_label,
                         int64_t nlabels, SymbolTable *syms) {
  for (int64_t i = 0; i < nlabels; ++i) {
    auto index = i + start_label;
    if (index != syms->AddSymbol(prefix + std::to_string(i), index)) {
      FSTERROR() << "AddAuxiliarySymbols: Symbol table clash";
      return false;
    }
  }
  return true;
}

}  // namespace fst
