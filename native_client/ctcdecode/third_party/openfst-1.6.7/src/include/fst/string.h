// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Utilities to convert strings into FSTs.

#ifndef FST_STRING_H_
#define FST_STRING_H_

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/compact-fst.h>
#include <fst/icu.h>
#include <fst/mutable-fst.h>
#include <fst/util.h>


DECLARE_string(fst_field_separator);

namespace fst {

enum StringTokenType { SYMBOL = 1, BYTE = 2, UTF8 = 3 };

namespace internal {

template <class Label>
bool ConvertSymbolToLabel(const char *str, const SymbolTable *syms,
                          Label unknown_label, bool allow_negative,
                          Label *output) {
  int64 n;
  if (syms) {
    n = syms->Find(str);
    if ((n == -1) && (unknown_label != kNoLabel)) n = unknown_label;
    if (n == -1 || (!allow_negative && n < 0)) {
      VLOG(1) << "ConvertSymbolToLabel: Symbol \"" << str
              << "\" is not mapped to any integer label, symbol table = "
              << syms->Name();
      return false;
    }
  } else {
    char *p;
    n = strtoll(str, &p, 10);
    if (p < str + strlen(str) || (!allow_negative && n < 0)) {
      VLOG(1) << "ConvertSymbolToLabel: Bad label integer "
              << "= \"" << str << "\"";
      return false;
    }
  }
  *output = n;
  return true;
}

template <class Label>
bool ConvertStringToLabels(const string &str, StringTokenType token_type,
                           const SymbolTable *syms, Label unknown_label,
                           bool allow_negative, std::vector<Label> *labels) {
  labels->clear();
  if (token_type == StringTokenType::BYTE) {
    for (const char c : str) labels->push_back(c);
  } else if (token_type == StringTokenType::UTF8) {
    return UTF8StringToLabels(str, labels);
  } else {
    std::unique_ptr<char[]> c_str(new char[str.size() + 1]);
    str.copy(c_str.get(), str.size());
    c_str[str.size()] = 0;
    std::vector<char *> vec;
    const string separator = "\n" + FLAGS_fst_field_separator;
    SplitString(c_str.get(), separator.c_str(), &vec, true);
    for (const char *c : vec) {
      Label label;
      if (!ConvertSymbolToLabel(c, syms, unknown_label, allow_negative,
                                &label)) {
        return false;
      }
      labels->push_back(label);
    }
  }
  return true;
}

}  // namespace internal

// Functor for compiling a string in an FST.
template <class Arc>
class StringCompiler {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit StringCompiler(StringTokenType token_type,
                          const SymbolTable *syms = nullptr,
                          Label unknown_label = kNoLabel,
                          bool allow_negative = false)
      : token_type_(token_type),
        syms_(syms),
        unknown_label_(unknown_label),
        allow_negative_(allow_negative) {}

  // Compiles string into an FST.
  template <class FST>
  bool operator()(const string &str, FST *fst) const {
    std::vector<Label> labels;
    if (!internal::ConvertStringToLabels(str, token_type_, syms_,
                                         unknown_label_, allow_negative_,
                                         &labels)) {
      return false;
    }
    Compile(labels, fst);
    return true;
  }

  template <class FST>
  bool operator()(const string &str, FST *fst, Weight weight) const {
    std::vector<Label> labels;
    if (!internal::ConvertStringToLabels(str, token_type_, syms_,
                                         unknown_label_, allow_negative_,
                                         &labels)) {
      return false;
    }
    Compile(labels, fst, std::move(weight));
    return true;
  }

 private:
  void Compile(const std::vector<Label> &labels, MutableFst<Arc> *fst,
               Weight weight = Weight::One()) const {
    fst->DeleteStates();
    while (fst->NumStates() <= labels.size()) fst->AddState();
    for (StateId i = 0; i < labels.size(); ++i) {
      fst->AddArc(i, Arc(labels[i], labels[i], Weight::One(), i + 1));
    }
    fst->SetStart(0);
    fst->SetFinal(labels.size(), std::move(weight));
  }

  template <class Unsigned>
  void Compile(const std::vector<Label> &labels,
               CompactStringFst<Arc, Unsigned> *fst) const {
    fst->SetCompactElements(labels.begin(), labels.end());
  }

  template <class Unsigned>
  void Compile(const std::vector<Label> &labels,
               CompactWeightedStringFst<Arc, Unsigned> *fst,
               const Weight &weight = Weight::One()) const {
    std::vector<std::pair<Label, Weight>> compacts;
    compacts.reserve(labels.size() + 1);
    for (StateId i = 0; i < static_cast<StateId>(labels.size()) - 1; ++i) {
      compacts.emplace_back(labels[i], Weight::One());
    }
    compacts.emplace_back(!labels.empty() ? labels.back() : kNoLabel, weight);
    fst->SetCompactElements(compacts.begin(), compacts.end());
  }

  const StringTokenType token_type_;
  const SymbolTable *syms_;    // Symbol table (used when token type is symbol).
  const Label unknown_label_;  // Label for token missing from symbol table.
  const bool allow_negative_;  // Negative labels allowed?

  StringCompiler(const StringCompiler &) = delete;
  StringCompiler &operator=(const StringCompiler &) = delete;
};

// Functor for printing a string FST as a string.
template <class Arc>
class StringPrinter {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit StringPrinter(StringTokenType token_type,
                         const SymbolTable *syms = nullptr)
      : token_type_(token_type), syms_(syms) {}

  // Converts the FST into a string.
  bool operator()(const Fst<Arc> &fst, string *result) {
    if (!FstToLabels(fst)) {
      VLOG(1) << "StringPrinter::operator(): FST is not a string";
      return false;
    }
    result->clear();
    if (token_type_ == StringTokenType::SYMBOL) {
      std::stringstream sstrm;
      for (size_t i = 0; i < labels_.size(); ++i) {
        if (i) sstrm << *(FLAGS_fst_field_separator.rbegin());
        if (!PrintLabel(labels_[i], sstrm)) return false;
      }
      *result = sstrm.str();
    } else if (token_type_ == StringTokenType::BYTE) {
      result->reserve(labels_.size());
      for (size_t i = 0; i < labels_.size(); ++i) result->push_back(labels_[i]);
    } else if (token_type_ == StringTokenType::UTF8) {
      return LabelsToUTF8String(labels_, result);
    } else {
      VLOG(1) << "StringPrinter::operator(): Unknown token type: "
              << token_type_;
      return false;
    }
    return true;
  }

 private:
  bool FstToLabels(const Fst<Arc> &fst) {
    labels_.clear();
    auto s = fst.Start();
    if (s == kNoStateId) {
      VLOG(2) << "StringPrinter::FstToLabels: Invalid starting state for "
              << "string FST";
      return false;
    }
    while (fst.Final(s) == Weight::Zero()) {
      ArcIterator<Fst<Arc>> aiter(fst, s);
      if (aiter.Done()) {
        VLOG(2) << "StringPrinter::FstToLabels: String FST traversal does "
                << "not reach final state";
        return false;
      }
      const auto &arc = aiter.Value();
      labels_.push_back(arc.olabel);
      s = arc.nextstate;
      if (s == kNoStateId) {
        VLOG(2) << "StringPrinter::FstToLabels: Transition to invalid state";
        return false;
      }
      aiter.Next();
      if (!aiter.Done()) {
        VLOG(2) << "StringPrinter::FstToLabels: State with multiple "
                << "outgoing arcs found";
        return false;
      }
    }
    return true;
  }

  bool PrintLabel(Label label, std::ostream &ostrm) {
    if (syms_) {
      const auto symbol = syms_->Find(label);
      if (symbol == "") {
        VLOG(2) << "StringPrinter::PrintLabel: Integer " << label << " is not "
                << "mapped to any textual symbol, symbol table = "
                << syms_->Name();
        return false;
      }
      ostrm << symbol;
    } else {
      ostrm << label;
    }
    return true;
  }

  const StringTokenType token_type_;
  const SymbolTable *syms_;    // Symbol table (used when token type is symbol).
  std::vector<Label> labels_;  // Input FST labels.

  StringPrinter(const StringPrinter &) = delete;
  StringPrinter &operator=(const StringPrinter &) = delete;
};

}  // namespace fst

#endif  // FST_STRING_H_
