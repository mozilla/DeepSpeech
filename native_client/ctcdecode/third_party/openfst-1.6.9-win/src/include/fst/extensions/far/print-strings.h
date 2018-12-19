// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Outputs as strings the string FSTs in a finite-state archive.

#ifndef FST_EXTENSIONS_FAR_PRINT_STRINGS_H_
#define FST_EXTENSIONS_FAR_PRINT_STRINGS_H_

#include <iomanip>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/extensions/far/far.h>
#include <fstream>
#include <fst/shortest-distance.h>
#include <fst/string.h>

DECLARE_string(far_field_separator);

namespace fst {

template <class Arc>
void FarPrintStrings(const std::vector<string> &ifilenames,
                     FarEntryType entry_type, FarTokenType far_token_type,
                     const string &begin_key, const string &end_key,
                     bool print_key, bool print_weight,
                     const string &symbols_fname, bool initial_symbols,
                     int32_t generate_filenames, const string &filename_prefix,
                     const string &filename_suffix) {
  StringTokenType token_type;
  if (far_token_type == FTT_SYMBOL) {
    token_type = StringTokenType::SYMBOL;
  } else if (far_token_type == FTT_BYTE) {
    token_type = StringTokenType::BYTE;
  } else if (far_token_type == FTT_UTF8) {
    token_type = StringTokenType::UTF8;
  } else {
    FSTERROR() << "FarPrintStrings: Unknown token type";
    return;
  }
  std::unique_ptr<const SymbolTable> syms;
  if (!symbols_fname.empty()) {
    // TODO(kbg): Allow negative flag?
    const SymbolTableTextOptions opts(true);
    syms.reset(SymbolTable::ReadText(symbols_fname, opts));
    if (!syms) {
      LOG(ERROR) << "FarPrintStrings: Error reading symbol table "
                 << symbols_fname;
      return;
    }
  }
  std::unique_ptr<FarReader<Arc>> far_reader(FarReader<Arc>::Open(ifilenames));
  if (!far_reader) return;
  if (!begin_key.empty()) far_reader->Find(begin_key);
  string okey;
  int nrep = 0;
  for (int i = 1; !far_reader->Done(); far_reader->Next(), ++i) {
    const auto &key = far_reader->GetKey();
    if (!end_key.empty() && end_key < key) break;
    if (okey == key) {
      ++nrep;
    } else {
      nrep = 0;
    }
    okey = key;
    const auto *fst = far_reader->GetFst();
    if (i == 1 && initial_symbols && !syms && fst->InputSymbols())
      syms.reset(fst->InputSymbols()->Copy());
    string str;
    VLOG(2) << "Handling key: " << key;
    StringPrinter<Arc> string_printer(token_type,
                                      syms ? syms.get() : fst->InputSymbols());
    string_printer(*fst, &str);
    if (entry_type == FET_LINE) {
      if (print_key) std::cout << key << FLAGS_far_field_separator[0];
      std::cout << str;
      if (print_weight)
        std::cout << FLAGS_far_field_separator[0] << ShortestDistance(*fst);
      std::cout << std::endl;
    } else if (entry_type == FET_FILE) {
      std::stringstream sstrm;
      if (generate_filenames) {
        sstrm.fill('0');
        sstrm << std::right << std::setw(generate_filenames) << i;
      } else {
        sstrm << key;
        if (nrep > 0) sstrm << "." << nrep;
      }
      string filename;
      filename = filename_prefix + sstrm.str() + filename_suffix;
      std::ofstream ostrm(filename);
      if (!ostrm) {
        LOG(ERROR) << "FarPrintStrings: Can't open file: " << filename;
        return;
      }
      ostrm << str;
      if (token_type == StringTokenType::SYMBOL) ostrm << "\n";
    }
  }
}

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_PRINT_STRINGS_H_
