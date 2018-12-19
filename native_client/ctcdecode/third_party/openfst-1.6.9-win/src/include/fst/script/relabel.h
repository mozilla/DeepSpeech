// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RELABEL_H_
#define FST_SCRIPT_RELABEL_H_

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include <fst/relabel.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using RelabelArgs1 = std::tuple<MutableFstClass *, const SymbolTable *,
                                const SymbolTable *, const string &, bool,
                                const SymbolTable *, const SymbolTable *,
                                const string &, bool>;

template <class Arc>
void Relabel(RelabelArgs1 *args) {
  MutableFst<Arc> *ofst = std::get<0>(*args)->GetMutableFst<Arc>();
  Relabel(ofst, std::get<1>(*args), std::get<2>(*args), std::get<3>(*args),
          std::get<4>(*args), std::get<5>(*args), std::get<6>(*args),
          std::get<7>(*args), std::get<8>(*args));
}

using LabelPair = std::pair<int64_t, int64_t>;

using RelabelArgs2 = std::tuple<MutableFstClass *,
                                const std::vector<LabelPair> &,
                                const std::vector<LabelPair> &>;

template <class Arc>
void Relabel(RelabelArgs2 *args) {
  MutableFst<Arc> *ofst = std::get<0>(*args)->GetMutableFst<Arc>();
  using LabelPair = std::pair<typename Arc::Label, typename Arc::Label>;
  // In case the MutableFstClass::Label is not the same as Arc::Label,
  // make a copy.
  std::vector<LabelPair> typed_ipairs(std::get<1>(*args).size());
  std::copy(std::get<1>(*args).begin(), std::get<1>(*args).end(),
            typed_ipairs.begin());
  std::vector<LabelPair> typed_opairs(std::get<2>(*args).size());
  std::copy(std::get<2>(*args).begin(), std::get<2>(*args).end(),
            typed_opairs.begin());
  Relabel(ofst, typed_ipairs, typed_opairs);
}

void Relabel(MutableFstClass *ofst,
             const SymbolTable *old_isymbols, const SymbolTable *new_isymbols,
             const string &unknown_isymbol,  bool attach_new_isymbols,
             const SymbolTable *old_osymbols, const SymbolTable *new_osymbols,
             const string &unknown_osymbol, bool attach_new_osymbols);

void Relabel(MutableFstClass *ofst, const std::vector<LabelPair> &ipairs,
             const std::vector<LabelPair> &opairs);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RELABEL_H_
