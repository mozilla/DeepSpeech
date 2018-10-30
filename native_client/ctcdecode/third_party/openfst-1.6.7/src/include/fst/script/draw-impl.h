// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to draw a binary FST by producing a text file in dot format, a helper
// class to fstdraw.cc.

#ifndef FST_SCRIPT_DRAW_IMPL_H_
#define FST_SCRIPT_DRAW_IMPL_H_

#include <ostream>
#include <sstream>
#include <string>

#include <fst/fst.h>
#include <fst/util.h>
#include <fst/script/fst-class.h>

namespace fst {

// Print a binary FST in GraphViz textual format (helper class for fstdraw.cc).
// WARNING: Stand-alone use not recommend.
template <class Arc>
class FstDrawer {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  FstDrawer(const Fst<Arc> &fst, const SymbolTable *isyms,
            const SymbolTable *osyms, const SymbolTable *ssyms, bool accep,
            const string &title, float width, float height, bool portrait,
            bool vertical, float ranksep, float nodesep, int fontsize,
            int precision, const string &float_format, bool show_weight_one)
      : fst_(fst),
        isyms_(isyms),
        osyms_(osyms),
        ssyms_(ssyms),
        accep_(accep && fst.Properties(kAcceptor, true)),
        ostrm_(nullptr),
        title_(title),
        width_(width),
        height_(height),
        portrait_(portrait),
        vertical_(vertical),
        ranksep_(ranksep),
        nodesep_(nodesep),
        fontsize_(fontsize),
        precision_(precision),
        float_format_(float_format),
        show_weight_one_(show_weight_one) {}

  // Draws FST to an output buffer.
  void Draw(std::ostream *strm, const string &dest) {
    ostrm_ = strm;
    SetStreamState(ostrm_);
    dest_ = dest;
    StateId start = fst_.Start();
    if (start == kNoStateId) return;
    PrintString("digraph FST {\n");
    if (vertical_) {
      PrintString("rankdir = BT;\n");
    } else {
      PrintString("rankdir = LR;\n");
    }
    PrintString("size = \"");
    Print(width_);
    PrintString(",");
    Print(height_);
    PrintString("\";\n");
    if (!dest_.empty()) PrintString("label = \"" + title_ + "\";\n");
    PrintString("center = 1;\n");
    if (portrait_) {
      PrintString("orientation = Portrait;\n");
    } else {
      PrintString("orientation = Landscape;\n");
    }
    PrintString("ranksep = \"");
    Print(ranksep_);
    PrintString("\";\n");
    PrintString("nodesep = \"");
    Print(nodesep_);
    PrintString("\";\n");
    // Initial state first.
    DrawState(start);
    for (StateIterator<Fst<Arc>> siter(fst_); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      if (s != start) DrawState(s);
    }
    PrintString("}\n");
  }

 private:
  void SetStreamState(std::ostream* strm) const {
    strm->precision(precision_);
    if (float_format_ == "e")
        strm->setf(std::ios_base::scientific, std::ios_base::floatfield);
    if (float_format_ == "f")
        strm->setf(std::ios_base::fixed, std::ios_base::floatfield);
    // O.w. defaults to "g" per standard lib.
  }

  void PrintString(const string &str) const { *ostrm_ << str; }

  // Escapes backslash and double quote if these occur in the string. Dot will
  // not deal gracefully with these if they are not escaped.
  static string Escape(const string &str) {
    string ns;
    for (char c : str) {
      if (c == '\\' || c == '"') ns.push_back('\\');
      ns.push_back(c);
    }
    return ns;
  }

  void PrintId(StateId id, const SymbolTable *syms, const char *name) const {
    if (syms) {
      auto symbol = syms->Find(id);
      if (symbol.empty()) {
        FSTERROR() << "FstDrawer: Integer " << id
                   << " is not mapped to any textual symbol"
                   << ", symbol table = " << syms->Name()
                   << ", destination = " << dest_;
        symbol = "?";
      }
      PrintString(Escape(symbol));
    } else {
      PrintString(std::to_string(id));
    }
  }

  void PrintStateId(StateId s) const { PrintId(s, ssyms_, "state ID"); }

  void PrintILabel(Label label) const {
    PrintId(label, isyms_, "arc input label");
  }

  void PrintOLabel(Label label) const {
    PrintId(label, osyms_, "arc output label");
  }

  void PrintWeight(Weight w) const {
    // Weight may have double quote characters in it, so escape it.
    PrintString(Escape(ToString(w)));
  }

  template <class T>
  void Print(T t) const { *ostrm_ << t; }

  template <class T>
  string ToString(T t) const {
    std::stringstream ss;
    SetStreamState(&ss);
    ss << t;
    return ss.str();
  }

  void DrawState(StateId s) const {
    Print(s);
    PrintString(" [label = \"");
    PrintStateId(s);
    const auto weight = fst_.Final(s);
    if (weight != Weight::Zero()) {
      if (show_weight_one_ || (weight != Weight::One())) {
        PrintString("/");
        PrintWeight(weight);
      }
      PrintString("\", shape = doublecircle,");
    } else {
      PrintString("\", shape = circle,");
    }
    if (s == fst_.Start()) {
      PrintString(" style = bold,");
    } else {
      PrintString(" style = solid,");
    }
    PrintString(" fontsize = ");
    Print(fontsize_);
    PrintString("]\n");
    for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      PrintString("\t");
      Print(s);
      PrintString(" -> ");
      Print(arc.nextstate);
      PrintString(" [label = \"");
      PrintILabel(arc.ilabel);
      if (!accep_) {
        PrintString(":");
        PrintOLabel(arc.olabel);
      }
      if (show_weight_one_ || (arc.weight != Weight::One())) {
        PrintString("/");
        PrintWeight(arc.weight);
      }
      PrintString("\", fontsize = ");
      Print(fontsize_);
      PrintString("];\n");
    }
  }

  const Fst<Arc> &fst_;
  const SymbolTable *isyms_;  // ilabel symbol table.
  const SymbolTable *osyms_;  // olabel symbol table.
  const SymbolTable *ssyms_;  // slabel symbol table.
  bool accep_;                // Print as acceptor when possible.
  std::ostream *ostrm_;       // Drawn FST destination.
  string dest_;               // Drawn FST destination name.

  string title_;
  float width_;
  float height_;
  bool portrait_;
  bool vertical_;
  float ranksep_;
  float nodesep_;
  int fontsize_;
  int precision_;
  string float_format_;
  bool show_weight_one_;

  FstDrawer(const FstDrawer &) = delete;
  FstDrawer &operator=(const FstDrawer &) = delete;
};

}  // namespace fst

#endif  // FST_SCRIPT_DRAW_IMPL_H_
