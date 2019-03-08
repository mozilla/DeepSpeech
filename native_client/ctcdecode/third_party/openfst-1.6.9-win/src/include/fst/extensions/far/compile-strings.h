// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_FAR_COMPILE_STRINGS_H_
#define FST_EXTENSIONS_FAR_COMPILE_STRINGS_H_

#ifndef _MSC_VER
#include <libgen.h>
#else
#include <fst/compat.h>
#endif

#include <fstream>
#include <istream>
#include <string>
#include <vector>

#include <fst/extensions/far/far.h>
#include <fstream>
#include <fst/string.h>

namespace fst {

// Constructs a reader that provides FSTs from a file (stream) either on a
// line-by-line basis or on a per-stream basis. Note that the freshly
// constructed reader is already set to the first input.
//
// Sample usage:
//
//   for (StringReader<Arc> reader(...); !reader.Done(); reader.Next()) {
//     auto *fst = reader.GetVectorFst();
//   }
template <class Arc>
class StringReader {
 public:
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;

  enum EntryType { LINE = 1, FILE = 2 };

  StringReader(std::istream &istrm, const string &source, EntryType entry_type,
               StringTokenType token_type, bool allow_negative_labels,
               const SymbolTable *syms = nullptr,
               Label unknown_label = kNoStateId)
      : nline_(0),
        istrm_(istrm),
        source_(source),
        entry_type_(entry_type),
        token_type_(token_type),
        symbols_(syms),
        done_(false),
        compiler_(token_type, syms, unknown_label, allow_negative_labels) {
    Next();  // Initialize the reader to the first input.
  }

  bool Done() { return done_; }

  void Next() {
    VLOG(1) << "Processing source " << source_ << " at line " << nline_;
    if (!istrm_) {  // We're done if we have no more input.
      done_ = true;
      return;
    }
    if (entry_type_ == LINE) {
      getline(istrm_, content_);
      ++nline_;
    } else {
      content_.clear();
      string line;
      while (getline(istrm_, line)) {
        ++nline_;
        content_.append(line);
        content_.append("\n");
      }
    }
    if (!istrm_ && content_.empty())  // We're also done if we read off all the
      done_ = true;                   // whitespace at the end of a file.
  }

  VectorFst<Arc> *GetVectorFst(bool keep_symbols = false) {
    std::unique_ptr<VectorFst<Arc>> fst(new VectorFst<Arc>());
    if (keep_symbols) {
      fst->SetInputSymbols(symbols_);
      fst->SetOutputSymbols(symbols_);
    }
    if (compiler_(content_, fst.get())) {
      return fst.release();
    } else {
      return nullptr;
    }
  }

  CompactStringFst<Arc> *GetCompactFst(bool keep_symbols = false) {
    std::unique_ptr<CompactStringFst<Arc>> fst;
    if (keep_symbols) {
      VectorFst<Arc> tmp;
      tmp.SetInputSymbols(symbols_);
      tmp.SetOutputSymbols(symbols_);
      fst.reset(new CompactStringFst<Arc>(tmp));
    } else {
      fst.reset(new CompactStringFst<Arc>());
    }
    if (compiler_(content_, fst.get())) {
      return fst.release();
    } else {
      return nullptr;
    }
  }

 private:
  size_t nline_;
  std::istream &istrm_;
  string source_;
  EntryType entry_type_;
  StringTokenType token_type_;
  const SymbolTable *symbols_;
  bool done_;
  StringCompiler<Arc> compiler_;
  string content_;  // The actual content of the input stream's next FST.

  StringReader(const StringReader &) = delete;
  StringReader &operator=(const StringReader &) = delete;
};

// Computes the minimal length required to encode each line number as a decimal
// number.
int KeySize(const char *filename);

template <class Arc>
void FarCompileStrings(const std::vector<string> &in_fnames,
                       const string &out_fname, const string &fst_type,
                       const FarType &far_type, int32_t generate_keys,
                       FarEntryType fet, FarTokenType tt,
                       const string &symbols_fname,
                       const string &unknown_symbol, bool keep_symbols,
                       bool initial_symbols, bool allow_negative_labels,
                       const string &key_prefix, const string &key_suffix) {
  typename StringReader<Arc>::EntryType entry_type;
  if (fet == FET_LINE) {
    entry_type = StringReader<Arc>::LINE;
  } else if (fet == FET_FILE) {
    entry_type = StringReader<Arc>::FILE;
  } else {
    FSTERROR() << "FarCompileStrings: Unknown entry type";
    return;
  }
  StringTokenType token_type;
  if (tt == FTT_SYMBOL) {
    token_type = StringTokenType::SYMBOL;
  } else if (tt == FTT_BYTE) {
    token_type = StringTokenType::BYTE;
  } else if (tt == FTT_UTF8) {
    token_type = StringTokenType::UTF8;
  } else {
    FSTERROR() << "FarCompileStrings: Unknown token type";
    return;
  }
  bool compact;
  if (fst_type.empty() || (fst_type == "vector")) {
    compact = false;
  } else if (fst_type == "compact") {
    compact = true;
  } else {
    FSTERROR() << "FarCompileStrings: Unknown FST type: " << fst_type;
    return;
  }
  std::unique_ptr<const SymbolTable> syms;
  typename Arc::Label unknown_label = kNoLabel;
  if (!symbols_fname.empty()) {
    const SymbolTableTextOptions opts(allow_negative_labels);
    syms.reset(SymbolTable::ReadText(symbols_fname, opts));
    if (!syms) {
      LOG(ERROR) << "FarCompileStrings: Error reading symbol table: "
                 << symbols_fname;
      return;
    }
    if (!unknown_symbol.empty()) {
      unknown_label = syms->Find(unknown_symbol);
      if (unknown_label == kNoLabel) {
        FSTERROR() << "FarCompileStrings: Label \"" << unknown_label
                   << "\" missing from symbol table: " << symbols_fname;
        return;
      }
    }
  }
  std::unique_ptr<FarWriter<Arc>> far_writer(
      FarWriter<Arc>::Create(out_fname, far_type));
  if (!far_writer) return;
  int n = 0;
  for (const auto &in_fname : in_fnames) {
    if (generate_keys == 0 && in_fname.empty()) {
      FSTERROR() << "FarCompileStrings: Read from a file instead of stdin or"
                 << " set the --generate_keys flags.";
      return;
    }
    int key_size =
        generate_keys ? generate_keys : (entry_type == StringReader<Arc>::FILE
                                             ? 1 : KeySize(in_fname.c_str()));
    std::ifstream fstrm;
    if (!in_fname.empty()) {
      fstrm.open(in_fname);
      if (!fstrm) {
        FSTERROR() << "FarCompileStrings: Can't open file: " << in_fname;
        return;
      }
    }
    std::istream &istrm = fstrm.is_open() ? fstrm : std::cin;
    bool keep_syms = keep_symbols;
    for (StringReader<Arc> reader(
             istrm, in_fname.empty() ? "stdin" : in_fname, entry_type,
             token_type, allow_negative_labels, syms.get(), unknown_label);
         !reader.Done(); reader.Next()) {
      ++n;
      std::unique_ptr<const Fst<Arc>> fst;
      if (compact) {
        fst.reset(reader.GetCompactFst(keep_syms));
      } else {
        fst.reset(reader.GetVectorFst(keep_syms));
      }
      if (initial_symbols) keep_syms = false;
      if (!fst) {
        FSTERROR() << "FarCompileStrings: Compiling string number " << n
                   << " in file " << in_fname << " failed with token_type = "
                   << (tt == FTT_BYTE
                           ? "byte"
                           : (tt == FTT_UTF8
                                  ? "utf8"
                                  : (tt == FTT_SYMBOL ? "symbol" : "unknown")))
                   << " and entry_type = "
                   << (fet == FET_LINE
                           ? "line"
                           : (fet == FET_FILE ? "file" : "unknown"));
        return;
      }
      std::ostringstream keybuf;
      keybuf.width(key_size);
      keybuf.fill('0');
      keybuf << n;
      string key;
      if (generate_keys > 0) {
        key = keybuf.str();
      } else {
        auto *filename = new char[in_fname.size() + 1];
        strcpy(filename, in_fname.c_str());
        key = basename(filename);
        if (entry_type != StringReader<Arc>::FILE) {
          key += "-";
          key += keybuf.str();
        }
        delete[] filename;
      }
      far_writer->Add(key_prefix + key + key_suffix, *fst);
    }
    if (generate_keys == 0) n = 0;
  }
}

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_COMPILE_STRINGS_H_
