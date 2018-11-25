// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Stand-alone class to print out binary FSTs in the AT&T format, a helper
// class for fstprint.cc.

#ifndef FST_SCRIPT_PRINT_IMPL_H_
#define FST_SCRIPT_PRINT_IMPL_H_

#include <ostream>
#include <sstream>
#include <string>

#include <fst/fstlib.h>
#include <fst/util.h>

DECLARE_string(fst_field_separator);

namespace fst {

// Print a binary FST in textual format (helper class for fstprint.cc).
// WARNING: Stand-alone use of this class not recommended, most code should
// read/write using the binary format which is much more efficient.
template <class Arc>
class FstPrinter {
 public:
  using StateId = typename Arc::StateId;
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;

  FstPrinter(const Fst<Arc> &fst, const SymbolTable *isyms,
             const SymbolTable *osyms, const SymbolTable *ssyms, bool accep,
             bool show_weight_one, const string &field_separator,
             const string &missing_symbol = "")
      : fst_(fst),
        isyms_(isyms),
        osyms_(osyms),
        ssyms_(ssyms),
        accep_(accep && fst.Properties(kAcceptor, true)),
        ostrm_(nullptr),
        show_weight_one_(show_weight_one),
        sep_(field_separator),
        missing_symbol_(missing_symbol) {}

  // Prints FST to an output stream.
  void Print(std::ostream *ostrm, const string &dest) {
    ostrm_ = ostrm;
    dest_ = dest;
    const auto start = fst_.Start();
    if (start == kNoStateId) return;
    // Initial state first.
    PrintState(start);
    for (StateIterator<Fst<Arc>> siter(fst_); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      if (s != start) PrintState(s);
    }
  }

 private:
  void PrintId(StateId id, const SymbolTable *syms, const char *name) const {
    if (syms) {
      string symbol = syms->Find(id);
      if (symbol.empty()) {
        if (missing_symbol_.empty()) {
          FSTERROR() << "FstPrinter: Integer " << id
                     << " is not mapped to any textual symbol"
                     << ", symbol table = " << syms->Name()
                     << ", destination = " << dest_;
          symbol = "?";
        } else {
          symbol = missing_symbol_;
        }
      }
      *ostrm_ << symbol;
    } else {
      *ostrm_ << id;
    }
  }

  void PrintStateId(StateId s) const { PrintId(s, ssyms_, "state ID"); }

  void PrintILabel(Label l) const { PrintId(l, isyms_, "arc input label"); }

  void PrintOLabel(Label l) const { PrintId(l, osyms_, "arc output label"); }

  void PrintState(StateId s) const {
    bool output = false;
    for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      PrintStateId(s);
      *ostrm_ << sep_;
      PrintStateId(arc.nextstate);
      *ostrm_ << sep_;
      PrintILabel(arc.ilabel);
      if (!accep_) {
        *ostrm_ << sep_;
        PrintOLabel(arc.olabel);
      }
      if (show_weight_one_ || arc.weight != Weight::One())
        *ostrm_ << sep_ << arc.weight;
      *ostrm_ << "\n";
      output = true;
    }
    const auto weight = fst_.Final(s);
    if (weight != Weight::Zero() || !output) {
      PrintStateId(s);
      if (show_weight_one_ || weight != Weight::One()) {
        *ostrm_ << sep_ << weight;
      }
      *ostrm_ << "\n";
    }
  }

  const Fst<Arc> &fst_;
  const SymbolTable *isyms_;  // ilabel symbol table.
  const SymbolTable *osyms_;  // olabel symbol table.
  const SymbolTable *ssyms_;  // slabel symbol table.
  bool accep_;                // Print as acceptor when possible?
  std::ostream *ostrm_;       // Text FST destination.
  string dest_;               // Text FST destination name.
  bool show_weight_one_;      // Print weights equal to Weight::One()?
  string sep_;                // Separator character between fields.
  string missing_symbol_;     // Symbol to print when lookup fails (default
                              // "" means raise error).
                              //
  FstPrinter(const FstPrinter &) = delete;
  FstPrinter &operator=(const FstPrinter &) = delete;
};

}  // namespace fst

#endif  // FST_SCRIPT_PRINT_IMPL_H_
