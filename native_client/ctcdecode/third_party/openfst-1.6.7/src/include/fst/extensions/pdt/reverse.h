// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Expands a PDT to an FST.

#ifndef FST_EXTENSIONS_PDT_REVERSE_H_
#define FST_EXTENSIONS_PDT_REVERSE_H_

#include <vector>

#include <fst/mutable-fst.h>
#include <fst/relabel.h>
#include <fst/reverse.h>

namespace fst {

// Reverses a pushdown transducer (PDT) encoded as an FST.
template <class Arc, class RevArc>
void Reverse(const Fst<Arc> &ifst,
             const std::vector<
                 std::pair<typename Arc::Label, typename Arc::Label>> &parens,
             MutableFst<RevArc> *ofst) {
  using Label = typename Arc::Label;
  // Reverses FST component.
  Reverse(ifst, ofst);
  // Exchanges open and close parenthesis pairs.
  std::vector<std::pair<Label, Label>> relabel_pairs;
  relabel_pairs.reserve(2 * parens.size());
  for (const auto &pair : parens) {
    relabel_pairs.emplace_back(pair.first, pair.second);
    relabel_pairs.emplace_back(pair.second, pair.first);
  }
  Relabel(ofst, relabel_pairs, relabel_pairs);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_REVERSE_H_
