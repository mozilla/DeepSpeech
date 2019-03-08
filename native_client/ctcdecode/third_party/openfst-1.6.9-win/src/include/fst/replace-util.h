// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Utility classes for the recursive replacement of FSTs (RTNs).

#ifndef FST_REPLACE_UTIL_H_
#define FST_REPLACE_UTIL_H_

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fst/log.h>

#include <fst/connect.h>
#include <fst/mutable-fst.h>
#include <fst/topsort.h>
#include <fst/vector-fst.h>


namespace fst {

// This specifies what labels to output on the call or return arc. Note that
// REPLACE_LABEL_INPUT and REPLACE_LABEL_OUTPUT will produce transducers when
// applied to acceptors.
enum ReplaceLabelType {
  // Epsilon labels on both input and output.
  REPLACE_LABEL_NEITHER = 1,
  // Non-epsilon labels on input and epsilon on output.
  REPLACE_LABEL_INPUT = 2,
  // Epsilon on input and non-epsilon on output.
  REPLACE_LABEL_OUTPUT = 3,
  // Non-epsilon labels on both input and output.
  REPLACE_LABEL_BOTH = 4
};

// By default ReplaceUtil will copy the input label of the replace arc.
// The call_label_type and return_label_type options specify how to manage
// the labels of the call arc and the return arc of the replace FST
struct ReplaceUtilOptions {
  int64_t root;                          // Root rule for expansion.
  ReplaceLabelType call_label_type;    // How to label call arc.
  ReplaceLabelType return_label_type;  // How to label return arc.
  int64_t return_label;                  // Label to put on return arc.

  explicit ReplaceUtilOptions(
      int64_t root = kNoLabel,
      ReplaceLabelType call_label_type = REPLACE_LABEL_INPUT,
      ReplaceLabelType return_label_type = REPLACE_LABEL_NEITHER,
      int64_t return_label = 0)
      : root(root),
        call_label_type(call_label_type),
        return_label_type(return_label_type),
        return_label(return_label) {}

  // For backwards compatibility.
  ReplaceUtilOptions(int64_t root, bool epsilon_replace_arc)
      : ReplaceUtilOptions(root,
                           epsilon_replace_arc ? REPLACE_LABEL_NEITHER
                                               : REPLACE_LABEL_INPUT) {}
};

// Every non-terminal on a path appears as the first label on that path in every
// FST associated with a given SCC of the replace dependency graph. This would
// be true if the SCC were formed from left-linear grammar rules.
constexpr uint8_t kReplaceSCCLeftLinear = 0x01;
// Every non-terminal on a path appears as the final label on that path in every
// FST associated with a given SCC of the replace dependency graph. This would
// be true if the SCC were formed from right-linear grammar rules.
constexpr uint8_t kReplaceSCCRightLinear = 0x02;
// The SCC in the replace dependency graph has more than one state or a
// self-loop.
constexpr uint8_t kReplaceSCCNonTrivial = 0x04;

// Defined in replace.h.
template <class Arc>
void Replace(
    const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>> &,
    MutableFst<Arc> *, const ReplaceUtilOptions &);

// Utility class for the recursive replacement of FSTs (RTNs). The user provides
// a set of label/FST pairs at construction. These are used by methods for
// testing cyclic dependencies and connectedness and doing RTN connection and
// specific FST replacement by label or for various optimization properties. The
// modified results can be obtained with the GetFstPairs() or
// GetMutableFstPairs() methods.
template <class Arc>
class ReplaceUtil {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstPair = std::pair<Label, const Fst<Arc> *>;
  using MutableFstPair = std::pair<Label, MutableFst<Arc> *>;
  using NonTerminalHash = std::unordered_map<Label, Label>;

  // Constructs from mutable FSTs; FST ownership is given to ReplaceUtil.
  ReplaceUtil(const std::vector<MutableFstPair> &fst_pairs,
              const ReplaceUtilOptions &opts);

  // Constructs from FSTs; FST ownership is retained by caller.
  ReplaceUtil(const std::vector<FstPair> &fst_pairs,
              const ReplaceUtilOptions &opts);

  // Constructs from ReplaceFst internals; FST ownership is retained by caller.
  ReplaceUtil(const std::vector<std::unique_ptr<const Fst<Arc>>> &fst_array,
              const NonTerminalHash &nonterminal_hash,
              const ReplaceUtilOptions &opts);

  ~ReplaceUtil() {
    for (Label i = 0; i < fst_array_.size(); ++i) delete fst_array_[i];
  }

  // True if the non-terminal dependencies are cyclic. Cyclic dependencies will
  // result in an unexpandable FST.
  bool CyclicDependencies() const {
    GetDependencies(false);
    return depprops_ & kCyclic;
  }

  // Returns the strongly-connected component ID in the dependency graph of the
  // replace FSTS.
  StateId SCC(Label label) const {
    GetDependencies(false);
    const auto it = nonterminal_hash_.find(label);
    if (it == nonterminal_hash_.end()) return kNoStateId;
    return depscc_[it->second];
  }

  // Returns properties for the strongly-connected component in the dependency
  // graph of the replace FSTs. If the SCC is kReplaceSCCLeftLinear or
  // kReplaceSCCRightLinear, that SCC can be represented as finite-state despite
  // any cyclic dependencies, but not by the usual replacement operation (see
  // fst/extensions/pdt/replace.h).
  uint8_t SCCProperties(StateId scc_id) {
    GetSCCProperties();
    return depsccprops_[scc_id];
  }

  // Returns true if no useless FSTs, states or transitions are present in the
  // RTN.
  bool Connected() const {
    GetDependencies(false);
    uint64_t props = kAccessible | kCoAccessible;
    for (Label i = 0; i < fst_array_.size(); ++i) {
      if (!fst_array_[i]) continue;
      if (fst_array_[i]->Properties(props, true) != props || !depaccess_[i]) {
        return false;
      }
    }
    return true;
  }

  // Removes useless FSTs, states and transitions from the RTN.
  void Connect();

  // Replaces FSTs specified by labels, unless there are cyclic dependencies.
  void ReplaceLabels(const std::vector<Label> &labels);

  // Replaces FSTs that have at most nstates states, narcs arcs and nnonterm
  // non-terminals (updating in reverse dependency order), unless there are
  // cyclic dependencies.
  void ReplaceBySize(size_t nstates, size_t narcs, size_t nnonterms);

  // Replaces singleton FSTS, unless there are cyclic dependencies.
  void ReplaceTrivial() { ReplaceBySize(2, 1, 1); }

  // Replaces non-terminals that have at most ninstances instances (updating in
  // dependency order), unless there are cyclic dependencies.
  void ReplaceByInstances(size_t ninstances);

  // Replaces non-terminals that have only one instance, unless there are cyclic
  // dependencies.
  void ReplaceUnique() { ReplaceByInstances(1); }

  // Returns label/FST pairs, retaining FST ownership.
  void GetFstPairs(std::vector<FstPair> *fst_pairs);

  // Returns label/mutable FST pairs, giving FST ownership over to the caller.
  void GetMutableFstPairs(std::vector<MutableFstPair> *mutable_fst_pairs);

 private:
  // FST statistics.
  struct ReplaceStats {
    StateId nstates;  // Number of states.
    StateId nfinal;   // Number of final states.
    size_t narcs;     // Number of arcs.
    Label nnonterms;  // Number of non-terminals in FST.
    size_t nref;      // Number of non-terminal instances referring to this FST.
    // Number of times that ith FST references this FST
    std::map<Label, size_t> inref;
    // Number of times that this FST references the ith FST
    std::map<Label, size_t> outref;

    ReplaceStats() : nstates(0), nfinal(0), narcs(0), nnonterms(0), nref(0) {}
  };

  // Checks that Mutable FSTs exists, creating them if necessary.
  void CheckMutableFsts();

  // Computes the dependency graph for the RTN, computing dependency statistics
  // if stats is true.
  void GetDependencies(bool stats) const;

  void ClearDependencies() const {
    depfst_.DeleteStates();
    stats_.clear();
    depprops_ = 0;
    depsccprops_.clear();
    have_stats_ = false;
  }

  // Gets topological order of dependencies, returning false with cyclic input.
  bool GetTopOrder(const Fst<Arc> &fst, std::vector<Label> *toporder) const;

  // Updates statistics to reflect the replacement of the jth FST.
  void UpdateStats(Label j);

  // Computes the properties for the strongly-connected component in the
  // dependency graph of the replace FSTs.
  void GetSCCProperties() const;

  Label root_label_;                                  // Root non-terminal.
  Label root_fst_;                                    // Root FST ID.
  ReplaceLabelType call_label_type_;                  // See Replace().
  ReplaceLabelType return_label_type_;                // See Replace().
  int64_t return_label_;                                // See Replace().
  std::vector<const Fst<Arc> *> fst_array_;           // FST per ID.
  std::vector<MutableFst<Arc> *> mutable_fst_array_;  // Mutable FST per ID.
  std::vector<Label> nonterminal_array_;              // FST ID to non-terminal.
  NonTerminalHash nonterminal_hash_;                  // Non-terminal to FST ID.
  mutable VectorFst<Arc> depfst_;                     // FST ID dependencies.
  mutable std::vector<StateId> depscc_;               // FST SCC ID.
  mutable std::vector<bool> depaccess_;               // FST ID accessibility.
  mutable uint64_t depprops_;                           // Dependency FST props.
  mutable bool have_stats_;                  // Have dependency statistics?
  mutable std::vector<ReplaceStats> stats_;  // Per-FST statistics.
  mutable std::vector<uint8_t> depsccprops_;   // SCC properties.
  ReplaceUtil(const ReplaceUtil &) = delete;
  ReplaceUtil &operator=(const ReplaceUtil &) = delete;
};

template <class Arc>
ReplaceUtil<Arc>::ReplaceUtil(const std::vector<MutableFstPair> &fst_pairs,
                              const ReplaceUtilOptions &opts)
    : root_label_(opts.root),
      call_label_type_(opts.call_label_type),
      return_label_type_(opts.return_label_type),
      return_label_(opts.return_label),
      depprops_(0),
      have_stats_(false) {
  fst_array_.push_back(nullptr);
  mutable_fst_array_.push_back(nullptr);
  nonterminal_array_.push_back(kNoLabel);
  for (const auto &fst_pair : fst_pairs) {
    const auto label = fst_pair.first;
    auto *fst = fst_pair.second;
    nonterminal_hash_[label] = fst_array_.size();
    nonterminal_array_.push_back(label);
    fst_array_.push_back(fst);
    mutable_fst_array_.push_back(fst);
  }
  root_fst_ = nonterminal_hash_[root_label_];
  if (!root_fst_) {
    FSTERROR() << "ReplaceUtil: No root FST for label: " << root_label_;
  }
}

template <class Arc>
ReplaceUtil<Arc>::ReplaceUtil(const std::vector<FstPair> &fst_pairs,
                              const ReplaceUtilOptions &opts)
    : root_label_(opts.root),
      call_label_type_(opts.call_label_type),
      return_label_type_(opts.return_label_type),
      return_label_(opts.return_label),
      depprops_(0),
      have_stats_(false) {
  fst_array_.push_back(nullptr);
  nonterminal_array_.push_back(kNoLabel);
  for (const auto &fst_pair : fst_pairs) {
    const auto label = fst_pair.first;
    const auto *fst = fst_pair.second;
    nonterminal_hash_[label] = fst_array_.size();
    nonterminal_array_.push_back(label);
    fst_array_.push_back(fst->Copy());
  }
  root_fst_ = nonterminal_hash_[root_label_];
  if (!root_fst_) {
    FSTERROR() << "ReplaceUtil: No root FST for label: " << root_label_;
  }
}

template <class Arc>
ReplaceUtil<Arc>::ReplaceUtil(
    const std::vector<std::unique_ptr<const Fst<Arc>>> &fst_array,
    const NonTerminalHash &nonterminal_hash, const ReplaceUtilOptions &opts)
    : root_fst_(opts.root),
      call_label_type_(opts.call_label_type),
      return_label_type_(opts.return_label_type),
      return_label_(opts.return_label),
      nonterminal_array_(fst_array.size()),
      nonterminal_hash_(nonterminal_hash),
      depprops_(0),
      have_stats_(false) {
  fst_array_.push_back(nullptr);
  for (size_t i = 1; i < fst_array.size(); ++i) {
    fst_array_.push_back(fst_array[i]->Copy());
  }
  for (auto it = nonterminal_hash.begin(); it != nonterminal_hash.end(); ++it) {
    nonterminal_array_[it->second] = it->first;
  }
  root_label_ = nonterminal_array_[root_fst_];
}

template <class Arc>
void ReplaceUtil<Arc>::GetDependencies(bool stats) const {
  if (depfst_.NumStates() > 0) {
    if (stats && !have_stats_) {
      ClearDependencies();
    } else {
      return;
    }
  }
  have_stats_ = stats;
  if (have_stats_) stats_.reserve(fst_array_.size());
  for (Label i = 0; i < fst_array_.size(); ++i) {
    depfst_.AddState();
    depfst_.SetFinal(i, Weight::One());
    if (have_stats_) stats_.push_back(ReplaceStats());
  }
  depfst_.SetStart(root_fst_);
  // An arc from each state (representing the FST) to the state representing the
  // FST being replaced
  for (Label i = 0; i < fst_array_.size(); ++i) {
    const auto *ifst = fst_array_[i];
    if (!ifst) continue;
    for (StateIterator<Fst<Arc>> siter(*ifst); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      if (have_stats_) {
        ++stats_[i].nstates;
        if (ifst->Final(s) != Weight::Zero()) ++stats_[i].nfinal;
      }
      for (ArcIterator<Fst<Arc>> aiter(*ifst, s); !aiter.Done();
           aiter.Next()) {
        if (have_stats_) ++stats_[i].narcs;
        const auto &arc = aiter.Value();
        auto it = nonterminal_hash_.find(arc.olabel);
        if (it != nonterminal_hash_.end()) {
          const auto j = it->second;
          depfst_.AddArc(i, Arc(arc.olabel, arc.olabel, Weight::One(), j));
          if (have_stats_) {
            ++stats_[i].nnonterms;
            ++stats_[j].nref;
            ++stats_[j].inref[i];
            ++stats_[i].outref[j];
          }
        }
      }
    }
  }
  // Computes accessibility info.
  SccVisitor<Arc> scc_visitor(&depscc_, &depaccess_, nullptr, &depprops_);
  DfsVisit(depfst_, &scc_visitor);
}

template <class Arc>
void ReplaceUtil<Arc>::UpdateStats(Label j) {
  if (!have_stats_) {
    FSTERROR() << "ReplaceUtil::UpdateStats: Stats not available";
    return;
  }
  if (j == root_fst_) return;  // Can't replace root.
  for (auto in = stats_[j].inref.begin(); in != stats_[j].inref.end(); ++in) {
    const auto i = in->first;
    const auto ni = in->second;
    stats_[i].nstates += stats_[j].nstates * ni;
    stats_[i].narcs += (stats_[j].narcs + 1) * ni;
    stats_[i].nnonterms += (stats_[j].nnonterms - 1) * ni;
    stats_[i].outref.erase(j);
    for (auto out = stats_[j].outref.begin(); out != stats_[j].outref.end();
         ++out) {
      const auto k = out->first;
      const auto nk = out->second;
      stats_[i].outref[k] += ni * nk;
    }
  }
  for (auto out = stats_[j].outref.begin(); out != stats_[j].outref.end();
       ++out) {
    const auto k = out->first;
    const auto nk = out->second;
    stats_[k].nref -= nk;
    stats_[k].inref.erase(j);
    for (auto in = stats_[j].inref.begin(); in != stats_[j].inref.end(); ++in) {
      const auto i = in->first;
      const auto ni = in->second;
      stats_[k].inref[i] += ni * nk;
      stats_[k].nref += ni * nk;
    }
  }
}

template <class Arc>
void ReplaceUtil<Arc>::CheckMutableFsts() {
  if (mutable_fst_array_.empty()) {
    for (Label i = 0; i < fst_array_.size(); ++i) {
      if (!fst_array_[i]) {
        mutable_fst_array_.push_back(nullptr);
      } else {
        mutable_fst_array_.push_back(new VectorFst<Arc>(*fst_array_[i]));
        delete fst_array_[i];
        fst_array_[i] = mutable_fst_array_[i];
      }
    }
  }
}

template <class Arc>
void ReplaceUtil<Arc>::Connect() {
  CheckMutableFsts();
  static constexpr auto props = kAccessible | kCoAccessible;
  for (auto *mutable_fst : mutable_fst_array_) {
    if (!mutable_fst) continue;
    if (mutable_fst->Properties(props, false) != props) {
      fst::Connect(mutable_fst);
    }
  }
  GetDependencies(false);
  for (Label i = 0; i < mutable_fst_array_.size(); ++i) {
    auto *fst = mutable_fst_array_[i];
    if (fst && !depaccess_[i]) {
      delete fst;
      fst_array_[i] = nullptr;
      mutable_fst_array_[i] = nullptr;
    }
  }
  ClearDependencies();
}

template <class Arc>
bool ReplaceUtil<Arc>::GetTopOrder(const Fst<Arc> &fst,
                                   std::vector<Label> *toporder) const {
  // Finds topological order of dependencies.
  std::vector<StateId> order;
  bool acyclic = false;
  TopOrderVisitor<Arc> top_order_visitor(&order, &acyclic);
  DfsVisit(fst, &top_order_visitor);
  if (!acyclic) {
    LOG(WARNING) << "ReplaceUtil::GetTopOrder: Cyclical label dependencies";
    return false;
  }
  toporder->resize(order.size());
  for (Label i = 0; i < order.size(); ++i) (*toporder)[order[i]] = i;
  return true;
}

template <class Arc>
void ReplaceUtil<Arc>::ReplaceLabels(const std::vector<Label> &labels) {
  CheckMutableFsts();
  std::unordered_set<Label> label_set;
  for (const auto label : labels) {
    // Can't replace root.
    if (label != root_label_) label_set.insert(label);
  }
  // Finds FST dependencies restricted to the labels requested.
  GetDependencies(false);
  VectorFst<Arc> pfst(depfst_);
  for (StateId i = 0; i < pfst.NumStates(); ++i) {
    std::vector<Arc> arcs;
    for (ArcIterator<VectorFst<Arc>> aiter(pfst, i); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto label = nonterminal_array_[arc.nextstate];
      if (label_set.count(label) > 0) arcs.push_back(arc);
    }
    pfst.DeleteArcs(i);
    for (const auto &arc : arcs) pfst.AddArc(i, arc);
  }
  std::vector<Label> toporder;
  if (!GetTopOrder(pfst, &toporder)) {
    ClearDependencies();
    return;
  }
  // Visits FSTs in reverse topological order of dependencies and performs
  // replacements.
  for (Label o = toporder.size() - 1; o >= 0; --o) {
    std::vector<FstPair> fst_pairs;
    auto s = toporder[o];
    for (ArcIterator<VectorFst<Arc>> aiter(pfst, s); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto label = nonterminal_array_[arc.nextstate];
      const auto *fst = fst_array_[arc.nextstate];
      fst_pairs.push_back(std::make_pair(label, fst));
    }
    if (fst_pairs.empty()) continue;
    const auto label = nonterminal_array_[s];
    const auto *fst = fst_array_[s];
    fst_pairs.push_back(std::make_pair(label, fst));
    const ReplaceUtilOptions opts(label, call_label_type_, return_label_type_,
                                  return_label_);
    Replace(fst_pairs, mutable_fst_array_[s], opts);
  }
  ClearDependencies();
}

template <class Arc>
void ReplaceUtil<Arc>::ReplaceBySize(size_t nstates, size_t narcs,
                                     size_t nnonterms) {
  std::vector<Label> labels;
  GetDependencies(true);
  std::vector<Label> toporder;
  if (!GetTopOrder(depfst_, &toporder)) {
    ClearDependencies();
    return;
  }
  for (Label o = toporder.size() - 1; o >= 0; --o) {
    const auto j = toporder[o];
    if (stats_[j].nstates <= nstates && stats_[j].narcs <= narcs &&
        stats_[j].nnonterms <= nnonterms) {
      labels.push_back(nonterminal_array_[j]);
      UpdateStats(j);
    }
  }
  ReplaceLabels(labels);
}

template <class Arc>
void ReplaceUtil<Arc>::ReplaceByInstances(size_t ninstances) {
  std::vector<Label> labels;
  GetDependencies(true);
  std::vector<Label> toporder;
  if (!GetTopOrder(depfst_, &toporder)) {
    ClearDependencies();
    return;
  }
  for (Label o = 0; o < toporder.size(); ++o) {
    const auto j = toporder[o];
    if (stats_[j].nref <= ninstances) {
      labels.push_back(nonterminal_array_[j]);
      UpdateStats(j);
    }
  }
  ReplaceLabels(labels);
}

template <class Arc>
void ReplaceUtil<Arc>::GetFstPairs(std::vector<FstPair> *fst_pairs) {
  CheckMutableFsts();
  fst_pairs->clear();
  for (Label i = 0; i < fst_array_.size(); ++i) {
    const auto label = nonterminal_array_[i];
    const auto *fst = fst_array_[i];
    if (!fst) continue;
    fst_pairs->push_back(std::make_pair(label, fst));
  }
}

template <class Arc>
void ReplaceUtil<Arc>::GetMutableFstPairs(
    std::vector<MutableFstPair> *mutable_fst_pairs) {
  CheckMutableFsts();
  mutable_fst_pairs->clear();
  for (Label i = 0; i < mutable_fst_array_.size(); ++i) {
    const auto label = nonterminal_array_[i];
    const auto *fst = mutable_fst_array_[i];
    if (!fst) continue;
    mutable_fst_pairs->push_back(std::make_pair(label, fst->Copy()));
  }
}

template <class Arc>
void ReplaceUtil<Arc>::GetSCCProperties() const {
  if (!depsccprops_.empty()) return;
  GetDependencies(false);
  if (depscc_.empty()) return;
  for (StateId scc = 0; scc < depscc_.size(); ++scc) {
    depsccprops_.push_back(kReplaceSCCLeftLinear | kReplaceSCCRightLinear);
  }
  if (!(depprops_ & kCyclic)) return;  // No cyclic dependencies.
  // Checks for self-loops in the dependency graph.
  for (StateId scc = 0; scc < depscc_.size(); ++scc) {
    for (ArcIterator<Fst<Arc> > aiter(depfst_, scc);
         !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      if (arc.nextstate == scc) {  // SCC has a self loop.
        depsccprops_[scc] |= kReplaceSCCNonTrivial;
      }
    }
  }
  std::vector<bool> depscc_visited(depscc_.size(), false);
  for (Label i = 0; i < fst_array_.size(); ++i) {
    const auto *fst = fst_array_[i];
    if (!fst) continue;
    const auto depscc = depscc_[i];
    if (depscc_visited[depscc]) {  // SCC has more than one state.
      depsccprops_[depscc] |= kReplaceSCCNonTrivial;
    }
    depscc_visited[depscc] = true;
    std::vector<StateId> fstscc;  // SCCs of the current FST.
    uint64_t fstprops;
    SccVisitor<Arc> scc_visitor(&fstscc, nullptr, nullptr, &fstprops);
    DfsVisit(*fst, &scc_visitor);
    for (StateIterator<Fst<Arc>> siter(*fst); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      for (ArcIterator<Fst<Arc>> aiter(*fst, s); !aiter.Done(); aiter.Next()) {
        const auto &arc = aiter.Value();
        auto it = nonterminal_hash_.find(arc.olabel);
        if (it == nonterminal_hash_.end() || depscc_[it->second] != depscc) {
          continue;  // Skips if a terminal or a non-terminal not in SCC.
        }
        const bool arc_in_cycle = fstscc[s] == fstscc[arc.nextstate];
        // Left linear iff all non-terminals are initial.
        if (s != fst->Start() || arc_in_cycle) {
          depsccprops_[depscc] &= ~kReplaceSCCLeftLinear;
        }
        // Right linear iff all non-terminals are final.
        if (fst->Final(arc.nextstate) == Weight::Zero() || arc_in_cycle) {
          depsccprops_[depscc] &= ~kReplaceSCCRightLinear;
        }
      }
    }
  }
}

}  // namespace fst

#endif  // FST_REPLACE_UTIL_H_
