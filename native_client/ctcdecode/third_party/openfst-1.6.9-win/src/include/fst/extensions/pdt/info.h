// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Prints information about a PDT.

#ifndef FST_EXTENSIONS_PDT_INFO_H_
#define FST_EXTENSIONS_PDT_INFO_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fst/extensions/pdt/pdt.h>
#include <fst/fst.h>

namespace fst {

// Compute various information about PDTs.
template <class Arc>
class PdtInfo {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  PdtInfo(const Fst<Arc> &fst,
          const std::vector<std::pair<Label, Label>> &parents);

  const string &FstType() const { return fst_type_; }

  const string &ArcType() const { return Arc::Type(); }

  int64_t NumStates() const { return nstates_; }

  int64_t NumArcs() const { return narcs_; }

  int64_t NumOpenParens() const { return nopen_parens_; }

  int64_t NumCloseParens() const { return nclose_parens_; }

  int64_t NumUniqueOpenParens() const { return nuniq_open_parens_; }

  int64_t NumUniqueCloseParens() const { return nuniq_close_parens_; }

  int64_t NumOpenParenStates() const { return nopen_paren_states_; }

  int64_t NumCloseParenStates() const { return nclose_paren_states_; }

 private:
  string fst_type_;
  int64_t nstates_;
  int64_t narcs_;
  int64_t nopen_parens_;
  int64_t nclose_parens_;
  int64_t nuniq_open_parens_;
  int64_t nuniq_close_parens_;
  int64_t nopen_paren_states_;
  int64_t nclose_paren_states_;
};

template <class Arc>
PdtInfo<Arc>::PdtInfo(
    const Fst<Arc> &fst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens)
    : fst_type_(fst.Type()),
      nstates_(0),
      narcs_(0),
      nopen_parens_(0),
      nclose_parens_(0),
      nuniq_open_parens_(0),
      nuniq_close_parens_(0),
      nopen_paren_states_(0),
      nclose_paren_states_(0) {
  std::unordered_map<Label, size_t> paren_map;
  std::unordered_set<Label> paren_set;
  std::unordered_set<StateId> open_paren_state_set;
  std::unordered_set<StateId> close_paren_state_set;
  for (size_t i = 0; i < parens.size(); ++i) {
    const auto &pair = parens[i];
    paren_map[pair.first] = i;
    paren_map[pair.second] = i;
  }
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    ++nstates_;
    const auto s = siter.Value();
    for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      ++narcs_;
      const auto it = paren_map.find(arc.ilabel);
      if (it != paren_map.end()) {
        const auto open_paren = parens[it->second].first;
        const auto close_paren = parens[it->second].second;
        if (arc.ilabel == open_paren) {
          ++nopen_parens_;
          if (!paren_set.count(open_paren)) {
            ++nuniq_open_parens_;
            paren_set.insert(open_paren);
          }
          if (!open_paren_state_set.count(arc.nextstate)) {
            ++nopen_paren_states_;
            open_paren_state_set.insert(arc.nextstate);
          }
        } else {
          ++nclose_parens_;
          if (!paren_set.count(close_paren)) {
            ++nuniq_close_parens_;
            paren_set.insert(close_paren);
          }
          if (!close_paren_state_set.count(s)) {
            ++nclose_paren_states_;
            close_paren_state_set.insert(s);
          }
        }
      }
    }
  }
}

template <class Arc>
void PrintPdtInfo(const PdtInfo<Arc> &info) {
  const auto old = std::cout.setf(std::ios::left);
  std::cout.width(50);
  std::cout << "fst type" << info.FstType() << std::endl;
  std::cout.width(50);
  std::cout << "arc type" << info.ArcType() << std::endl;
  std::cout.width(50);
  std::cout << "# of states" << info.NumStates() << std::endl;
  std::cout.width(50);
  std::cout << "# of arcs" << info.NumArcs() << std::endl;
  std::cout.width(50);
  std::cout << "# of open parentheses" << info.NumOpenParens() << std::endl;
  std::cout.width(50);
  std::cout << "# of close parentheses" << info.NumCloseParens() << std::endl;
  std::cout.width(50);
  std::cout << "# of unique open parentheses" << info.NumUniqueOpenParens()
            << std::endl;
  std::cout.width(50);
  std::cout << "# of unique close parentheses" << info.NumUniqueCloseParens()
            << std::endl;
  std::cout.width(50);
  std::cout << "# of open parenthesis dest. states" << info.NumOpenParenStates()
            << std::endl;
  std::cout.width(50);
  std::cout << "# of close parenthesis source states"
            << info.NumCloseParenStates() << std::endl;
  std::cout.setf(old);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_INFO_H_
