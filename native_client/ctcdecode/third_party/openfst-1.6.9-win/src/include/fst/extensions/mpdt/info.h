// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Prints information about an MPDT.

#ifndef FST_EXTENSIONS_MPDT_INFO_H_
#define FST_EXTENSIONS_MPDT_INFO_H_

#include <unordered_map>
#include <vector>

#include <fst/extensions/mpdt/mpdt.h>
#include <fst/fst.h>

namespace fst {

// Compute various information about MPDTs, helper class for mpdtinfo.cc.
template <class Arc, typename Arc::Label nlevels = 2>
class MPdtInfo {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  MPdtInfo(const Fst<Arc> &fst,
           const std::vector<std::pair<Label, Label>> &parens,
           const std::vector<Label> &assignments);

  const string &FstType() const { return fst_type_; }

  const string &ArcType() const { return Arc::Type(); }

  int64_t NumStates() const { return nstates_; }

  int64_t NumArcs() const { return narcs_; }

  int64_t NumLevels() const { return nlevels; }

  int64_t NumOpenParens(Label level) const { return nopen_parens_[level]; }

  int64_t NumCloseParens(Label level) const { return nclose_parens_[level]; }

  int64_t NumUniqueOpenParens(Label level) const {
    return nuniq_open_parens_[level];
  }

  int64_t NumUniqueCloseParens(Label level) const {
    return nuniq_close_parens_[level];
  }
  int64_t NumOpenParenStates(Label level) const {
    return nopen_paren_states_[level];
  }

  int64_t NumCloseParenStates(Label level) const {
    return nclose_paren_states_[level];
  }

  void Print();

 private:
  string fst_type_;
  int64_t nstates_;
  int64_t narcs_;
  int64_t nopen_parens_[nlevels];
  int64_t nclose_parens_[nlevels];
  int64_t nuniq_open_parens_[nlevels];
  int64_t nuniq_close_parens_[nlevels];
  int64_t nopen_paren_states_[nlevels];
  int64_t nclose_paren_states_[nlevels];
  bool error_;
};

template <class Arc, typename Arc::Label nlevels>
MPdtInfo<Arc, nlevels>::MPdtInfo(
    const Fst<Arc> &fst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &parens,
    const std::vector<typename Arc::Label> &assignments)
    : fst_type_(fst.Type()), nstates_(0), narcs_(0), error_(false) {
  std::unordered_map<Label, size_t> paren_map;
  std::unordered_set<Label> paren_set;
  std::unordered_map<Label, int> paren_levels;
  std::unordered_set<StateId> open_paren_state_set;
  std::unordered_set<StateId> close_paren_state_set;
  if (parens.size() != assignments.size()) {
    FSTERROR() << "MPdtInfo: Parens of different size from assignments";
    error_ = true;
    return;
  }
  for (Label i = 0; i < assignments.size(); ++i) {
    // Assignments here start at 0, so assuming the human-readable version has
    // them starting at 1, we should subtract 1 here.
    Label level = assignments[i] - 1;
    if (level < 0 || level >= nlevels) {
      FSTERROR() << "MPdtInfo: Specified level " << level << " out of bounds";
      error_ = true;
      return;
    }
    const auto &pair = parens[i];
    paren_levels[pair.first] = level;
    paren_levels[pair.second] = level;
    paren_map[pair.first] = i;
    paren_map[pair.second] = i;
  }
  for (Label i = 0; i < nlevels; ++i) {
    nopen_parens_[i] = 0;
    nclose_parens_[i] = 0;
    nuniq_open_parens_[i] = 0;
    nuniq_close_parens_[i] = 0;
    nopen_paren_states_[i] = 0;
    nclose_paren_states_[i] = 0;
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
        const auto level = paren_levels[arc.ilabel];
        if (arc.ilabel == open_paren) {
          ++nopen_parens_[level];
          if (!paren_set.count(open_paren)) {
            ++nuniq_open_parens_[level];
            paren_set.insert(open_paren);
          }
          if (!open_paren_state_set.count(arc.nextstate)) {
            ++nopen_paren_states_[level];
            open_paren_state_set.insert(arc.nextstate);
          }
        } else {
          ++nclose_parens_[level];
          if (!paren_set.count(close_paren)) {
            ++nuniq_close_parens_[level];
            paren_set.insert(close_paren);
          }
          if (!close_paren_state_set.count(s)) {
            ++nclose_paren_states_[level];
            close_paren_state_set.insert(s);
          }
        }
      }
    }
  }
}

template <class Arc, typename Arc::Label nlevels>
void MPdtInfo<Arc, nlevels>::Print() {
  const auto old = std::cout.setf(std::ios::left);
  std::cout.width(50);
  std::cout << "fst type" << FstType() << std::endl;
  std::cout.width(50);
  std::cout << "arc type" << ArcType() << std::endl;
  std::cout.width(50);
  std::cout << "# of states" << NumStates() << std::endl;
  std::cout.width(50);
  std::cout << "# of arcs" << NumArcs() << std::endl;
  std::cout.width(50);
  std::cout << "# of levels" << NumLevels() << std::endl;
  std::cout.width(50);
  for (typename Arc::Label i = 0; i < nlevels; ++i) {
    int level = i + 1;
    std::cout << "# of open parentheses at levelel " << level << "\t"
              << NumOpenParens(i) << std::endl;
    std::cout.width(50);
    std::cout << "# of close parentheses at levelel " << level << "\t"
              << NumCloseParens(i) << std::endl;
    std::cout.width(50);
    std::cout << "# of unique open parentheses at levelel " << level << "\t"
              << NumUniqueOpenParens(i) << std::endl;
    std::cout.width(50);
    std::cout << "# of unique close parentheses at levelel " << level << "\t"
              << NumUniqueCloseParens(i) << std::endl;
    std::cout.width(50);
    std::cout << "# of open parenthesis dest. states at levelel " << level
              << "\t" << NumOpenParenStates(i) << std::endl;
    std::cout.width(50);
    std::cout << "# of close parenthesis source states at levelel " << level
              << "\t" << NumCloseParenStates(i) << std::endl;
    std::cout.width(50);
  }
  std::cout.setf(old);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_MPDT_INFO_H_
