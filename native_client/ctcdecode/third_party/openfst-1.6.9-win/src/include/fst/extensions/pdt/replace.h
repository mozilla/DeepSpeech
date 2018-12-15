// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Recursively replaces FST arcs with other FSTs, returning a PDT.

#ifndef FST_EXTENSIONS_PDT_REPLACE_H_
#define FST_EXTENSIONS_PDT_REPLACE_H_

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/replace.h>
#include <fst/replace-util.h>
#include <fst/symbol-table-ops.h>

namespace fst {
namespace internal {

// Hash to paren IDs
template <typename S>
struct ReplaceParenHash {
  size_t operator()(const std::pair<size_t, S> &paren) const {
    static constexpr auto prime = 7853;
    return paren.first + paren.second * prime;
  }
};

}  // namespace internal

// Parser types characterize the PDT construction method. When applied to a CFG,
// each non-terminal is encoded as a DFA that accepts precisely the RHS's of
// productions of that non-terminal. For parsing (rather than just recognition),
// production numbers can used as outputs (placed as early as possible) in the
// DFAs promoted to DFTs. For more information on the strongly regular
// construction, see:
//
// Mohri, M., and Pereira, F. 1998. Dynamic compilation of weighted context-free
// grammars. In Proc. ACL, pages 891-897.
enum PdtParserType {
  // Top-down construction. Applied to a simple LL(1) grammar (among others),
  // gives a DPDA. If promoted to a DPDT, with outputs being production
  // numbers, gives a leftmost derivation. Left recursive grammars are
  // problematic in use.
  PDT_LEFT_PARSER,

  // Top-down construction. Similar to PDT_LEFT_PARSE except bounded-stack
  // (expandable as an FST) result with regular or, more generally, strongly
  // regular grammars. Epsilons may replace some parentheses, which may
  // introduce some non-determinism.
  PDT_LEFT_SR_PARSER,

  /* TODO(riley):
  // Bottom-up construction. Applied to a LR(0) grammar, gives a DPDA.
  // If promoted to a DPDT, with outputs being the production nubmers,
  // gives the reverse of a rightmost derivation.
  PDT_RIGHT_PARSER,
  */
};

template <class Arc>
struct PdtReplaceOptions {
  using Label = typename Arc::Label;

  explicit PdtReplaceOptions(Label root,
                             PdtParserType type = PDT_LEFT_PARSER,
                             Label start_paren_labels = kNoLabel,
                             string left_paren_prefix = "(_",
                             string right_paren_prefix = ")_") :
      root(root), type(type), start_paren_labels(start_paren_labels),
      left_paren_prefix(std::move(left_paren_prefix)),
      right_paren_prefix(std::move(right_paren_prefix)) {}

  Label root;
  PdtParserType type;
  Label start_paren_labels;
  const string left_paren_prefix;
  const string right_paren_prefix;
};

// PdtParser: Base PDT parser class common to specific parsers.

template <class Arc>
class PdtParser {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using LabelFstPair = std::pair<Label, const Fst<Arc> *>;
  using LabelPair = std::pair<Label, Label>;
  using LabelStatePair = std::pair<Label, StateId>;
  using StateWeightPair = std::pair<StateId, Weight>;
  using ParenKey = std::pair<size_t, StateId>;
  using ParenMap =
      std::unordered_map<ParenKey, size_t, internal::ReplaceParenHash<StateId>>;

  PdtParser(const std::vector<LabelFstPair> &fst_array,
            const PdtReplaceOptions<Arc> &opts) :
      root_(opts.root), start_paren_labels_(opts.start_paren_labels),
      left_paren_prefix_(std::move(opts.left_paren_prefix)),
      right_paren_prefix_(std::move(opts.right_paren_prefix)),
      error_(false) {
    for (size_t i = 0; i < fst_array.size(); ++i) {
      if (!CompatSymbols(fst_array[0].second->InputSymbols(),
                         fst_array[i].second->InputSymbols())) {
        FSTERROR() << "PdtParser: Input symbol table of input FST " << i
                   << " does not match input symbol table of 0th input FST";
        error_ = true;
      }
      if (!CompatSymbols(fst_array[0].second->OutputSymbols(),
                         fst_array[i].second->OutputSymbols())) {
        FSTERROR() << "PdtParser: Output symbol table of input FST " << i
                   << " does not match input symbol table of 0th input FST";
        error_ = true;
      }
      fst_array_.emplace_back(fst_array[i].first, fst_array[i].second->Copy());
      // Builds map from non-terminal label to FST ID.
      label2id_[fst_array[i].first] = i;
    }
  }

  virtual ~PdtParser() {
    for (auto &pair : fst_array_) delete pair.second;
  }

  // Constructs the output PDT, dependent on the derived parser type.
  virtual void GetParser(MutableFst<Arc> *ofst,
                         std::vector<LabelPair> *parens) = 0;

 protected:
  const std::vector<LabelFstPair> &FstArray() const { return fst_array_; }

  Label Root() const { return root_; }

  // Maps from non-terminal label to corresponding FST ID, or returns
  // kNoStateId to signal lookup failure.
  StateId Label2Id(Label l) const {
    auto it = label2id_.find(l);
    return it == label2id_.end() ? kNoStateId : it->second;
  }

  // Maps from output state to input FST label, state pair, or returns a
  // (kNoLabel, kNoStateId) pair to signal lookup failure.
  LabelStatePair GetLabelStatePair(StateId os) const {
    if (os >= label_state_pairs_.size()) {
      static const LabelStatePair no_pair(kNoLabel, kNoLabel);
      return no_pair;
    } else {
      return label_state_pairs_[os];
    }
  }

  // Maps to output state from input FST (label, state) pair, or returns
  // kNoStateId to signal lookup failure.
  StateId GetState(const LabelStatePair &lsp) const {
    auto it = state_map_.find(lsp);
    if (it == state_map_.end()) {
      return kNoStateId;
    } else {
      return it->second;
    }
  }

  // Builds single FST combining all referenced input FSTs, leaving in the
  // non-termnals for now; also tabulates the PDT states that correspond to the
  // start and final states of the input FSTs.
  void CreateFst(MutableFst<Arc> *ofst, std::vector<StateId> *open_dest,
                 std::vector<std::vector<StateWeightPair>> *close_src);

  // Assigns parenthesis labels from total allocated paren IDs.
  void AssignParenLabels(size_t total_nparens, std::vector<LabelPair> *parens) {
    parens->clear();
    for (size_t paren_id = 0; paren_id < total_nparens; ++paren_id) {
      const auto open_paren = start_paren_labels_ + paren_id;
      const auto close_paren = open_paren + total_nparens;
      parens->emplace_back(open_paren, close_paren);
    }
  }

  // Determines how non-terminal instances are assigned parentheses IDs.
  virtual size_t AssignParenIds(const Fst<Arc> &ofst,
                                ParenMap *paren_map) const = 0;

  // Changes a non-terminal transition to an open parenthesis transition
  // redirected to the PDT state specified in the open_dest argument, when
  // indexed by the input FST ID for the non-terminal. Adds close parenthesis
  // transitions (with specified weights) from the PDT states specified in the
  // close_src argument, when indexed by the input FST ID for the non-terminal,
  // to the former destination state of the non-terminal transition. The
  // paren_map argument gives the parenthesis ID for a given non-terminal FST ID
  // and destination state pair. The close_non_term_weight vector specifies
  // non-terminals for which the non-terminal arc weight should be applied on
  // the close parenthesis (multiplying the close_src weight above) rather than
  // on the open parenthesis. If no paren ID is found, then an epsilon replaces
  // the parenthesis that would carry the non-terminal arc weight and the other
  // parenthesis is omitted (appropriate for the strongly-regular case).
  void AddParensToFst(
      const std::vector<LabelPair> &parens,
      const ParenMap &paren_map,
      const std::vector<StateId> &open_dest,
      const std::vector<std::vector<StateWeightPair>> &close_src,
      const std::vector<bool> &close_non_term_weight,
      MutableFst<Arc> *ofst);

  // Ensures that parentheses arcs are added to the symbol table.
  void AddParensToSymbolTables(const std::vector<LabelPair> &parens,
                               MutableFst<Arc> *ofst);

 private:
  std::vector<LabelFstPair> fst_array_;
  Label root_;
  // Index to use for the first parenthesis.
  Label start_paren_labels_;
  const string left_paren_prefix_;
  const string right_paren_prefix_;
  // Maps from non-terminal label to FST ID.
  std::unordered_map<Label, StateId> label2id_;
  // Given an output state, specifies the input FST (label, state) pair.
  std::vector<LabelStatePair> label_state_pairs_;
  // Given an FST (label, state) pair, specifies the output FST state ID.
  std::map<LabelStatePair, StateId> state_map_;
  bool error_;
};

template <class Arc>
void PdtParser<Arc>::CreateFst(
    MutableFst<Arc> *ofst, std::vector<StateId> *open_dest,
    std::vector<std::vector<StateWeightPair>> *close_src) {
  ofst->DeleteStates();
  if (error_) {
    ofst->SetProperties(kError, kError);
    return;
  }
  open_dest->resize(fst_array_.size(), kNoStateId);
  close_src->resize(fst_array_.size());
  // Queue of non-terminals to replace.
  std::deque<Label> non_term_queue;
  non_term_queue.push_back(root_);
  // Has a non-terminal been enqueued?
  std::vector<bool> enqueued(fst_array_.size(), false);
  enqueued[label2id_[root_]] = true;
  Label max_label = kNoLabel;
  for (StateId soff = 0; !non_term_queue.empty(); soff = ofst->NumStates()) {
    const auto label = non_term_queue.front();
    non_term_queue.pop_front();
    StateId fst_id = Label2Id(label);
    const auto *ifst = fst_array_[fst_id].second;
    for (StateIterator<Fst<Arc>> siter(*ifst); !siter.Done(); siter.Next()) {
      const auto is = siter.Value();
      const auto os = ofst->AddState();
      const LabelStatePair lsp(label, is);
      label_state_pairs_.push_back(lsp);
      state_map_[lsp] = os;
      if (is == ifst->Start()) {
        (*open_dest)[fst_id] = os;
        if (label == root_) ofst->SetStart(os);
      }
      if (ifst->Final(is) != Weight::Zero()) {
        if (label == root_) ofst->SetFinal(os, ifst->Final(is));
        (*close_src)[fst_id].emplace_back(os, ifst->Final(is));
      }
      for (ArcIterator<Fst<Arc>> aiter(*ifst, is); !aiter.Done();
           aiter.Next()) {
        auto arc = aiter.Value();
        arc.nextstate += soff;
        if (max_label == kNoLabel || arc.olabel > max_label)
          max_label = arc.olabel;
        const auto nfst_id = Label2Id(arc.olabel);
        if (nfst_id != kNoStateId) {
          if (fst_array_[nfst_id].second->Start() == kNoStateId) continue;
          if (!enqueued[nfst_id]) {
            non_term_queue.push_back(arc.olabel);
            enqueued[nfst_id] = true;
          }
        }
        ofst->AddArc(os, arc);
      }
    }
  }
  if (start_paren_labels_ == kNoLabel)
    start_paren_labels_ = max_label + 1;
}

template <class Arc>
void PdtParser<Arc>::AddParensToFst(
    const std::vector<LabelPair> &parens,
    const ParenMap &paren_map,
    const std::vector<StateId> &open_dest,
    const std::vector<std::vector<StateWeightPair>> &close_src,
    const std::vector<bool> &close_non_term_weight,
    MutableFst<Arc> *ofst) {
  StateId dead_state = kNoStateId;
  using MIter = MutableArcIterator<MutableFst<Arc>>;
  for (StateIterator<Fst<Arc>> siter(*ofst); !siter.Done(); siter.Next()) {
    StateId os = siter.Value();
    std::unique_ptr<MIter> aiter(new MIter(ofst, os));
    for (auto n = 0; !aiter->Done(); aiter->Next(), ++n) {
      const auto arc = aiter->Value();  // A reference here may go stale.
      StateId nfst_id = Label2Id(arc.olabel);
      if (nfst_id != kNoStateId) {
        // Gets parentheses.
        const ParenKey paren_key(nfst_id, arc.nextstate);
        auto it = paren_map.find(paren_key);
        Label open_paren = 0;
        Label close_paren = 0;
        if (it != paren_map.end()) {
          const auto paren_id = it->second;
          open_paren = parens[paren_id].first;
          close_paren = parens[paren_id].second;
        }
        // Sets open parenthesis.
        if (open_paren != 0 || !close_non_term_weight[nfst_id]) {
          const auto open_weight =
              close_non_term_weight[nfst_id] ? Weight::One() : arc.weight;
          const Arc sarc(open_paren, open_paren, open_weight,
                         open_dest[nfst_id]);
          aiter->SetValue(sarc);
        } else {
          if (dead_state == kNoStateId) {
            dead_state = ofst->AddState();
          }
          const Arc sarc(0, 0, Weight::One(), dead_state);
          aiter->SetValue(sarc);
        }
        // Adds close parentheses.
        if (close_paren != 0 || close_non_term_weight[nfst_id]) {
          for (size_t i = 0; i < close_src[nfst_id].size(); ++i) {
            const auto &pair = close_src[nfst_id][i];
            const auto close_weight = close_non_term_weight[nfst_id]
                                          ? Times(arc.weight, pair.second)
                                          : pair.second;
            const Arc farc(close_paren, close_paren, close_weight,
                           arc.nextstate);

            ofst->AddArc(pair.first, farc);
            if (os == pair.first) {  // Invalidated iterator.
              aiter.reset(new MIter(ofst, os));
              aiter->Seek(n);
            }
          }
        }
      }
    }
  }
}

template <class Arc>
void PdtParser<Arc>::AddParensToSymbolTables(
    const std::vector<LabelPair> &parens, MutableFst<Arc> *ofst) {
  auto size = parens.size();
  if (ofst->InputSymbols()) {
    if (!AddAuxiliarySymbols(left_paren_prefix_, start_paren_labels_, size,
                             ofst->MutableInputSymbols())) {
      ofst->SetProperties(kError, kError);
      return;
    }
    if (!AddAuxiliarySymbols(right_paren_prefix_, start_paren_labels_ + size,
                             size, ofst->MutableInputSymbols())) {
      ofst->SetProperties(kError, kError);
      return;
    }
  }
  if (ofst->OutputSymbols()) {
    if (!AddAuxiliarySymbols(left_paren_prefix_, start_paren_labels_, size,
                             ofst->MutableOutputSymbols())) {
      ofst->SetProperties(kError, kError);
      return;
    }
    if (!AddAuxiliarySymbols(right_paren_prefix_, start_paren_labels_ + size,
                             size, ofst->MutableOutputSymbols())) {
      ofst->SetProperties(kError, kError);
      return;
    }
  }
}

// Builds a PDT by recursive replacement top-down, where the call and return are
// encoded in the parentheses.
template <class Arc>
class PdtLeftParser final : public PdtParser<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using LabelFstPair = typename PdtParser<Arc>::LabelFstPair;
  using LabelPair = typename PdtParser<Arc>::LabelPair;
  using LabelStatePair = typename PdtParser<Arc>::LabelStatePair;
  using StateWeightPair = typename PdtParser<Arc>::StateWeightPair;
  using ParenKey = typename PdtParser<Arc>::ParenKey;
  using ParenMap = typename PdtParser<Arc>::ParenMap;

  using PdtParser<Arc>::AddParensToFst;
  using PdtParser<Arc>::AddParensToSymbolTables;
  using PdtParser<Arc>::AssignParenLabels;
  using PdtParser<Arc>::CreateFst;
  using PdtParser<Arc>::FstArray;
  using PdtParser<Arc>::GetLabelStatePair;
  using PdtParser<Arc>::GetState;
  using PdtParser<Arc>::Label2Id;
  using PdtParser<Arc>::Root;

  PdtLeftParser(const std::vector<LabelFstPair> &fst_array,
                const PdtReplaceOptions<Arc> &opts) :
      PdtParser<Arc>(fst_array, opts) { }

  void GetParser(MutableFst<Arc> *ofst,
                 std::vector<LabelPair> *parens) override;

 protected:
  // Assigns a unique parenthesis ID for each non-terminal, destination
  // state pair.
  size_t AssignParenIds(const Fst<Arc> &ofst,
                        ParenMap *paren_map) const override;
};

template <class Arc>
void PdtLeftParser<Arc>::GetParser(
    MutableFst<Arc> *ofst,
    std::vector<LabelPair> *parens) {
  ofst->DeleteStates();
  parens->clear();
  const auto &fst_array = FstArray();
  // Map that gives the paren ID for a (non-terminal, dest. state) pair
  // (which can be unique).
  ParenMap paren_map;
  // Specifies the open parenthesis destination state for a given non-terminal.
  // The source is the non-terminal instance source state.
  std::vector<StateId> open_dest(fst_array.size(), kNoStateId);
  // Specifies close parenthesis source states and weights for a given
  // non-terminal. The destination is the non-terminal instance destination
  // state.
  std::vector<std::vector<StateWeightPair>> close_src(fst_array.size());
  // Specifies non-terminals for which the non-terminal arc weight
  // should be applied on the close parenthesis (multiplying the
  // 'close_src' weight above) rather than on the open parenthesis.
  std::vector<bool> close_non_term_weight(fst_array.size(), false);
  CreateFst(ofst, &open_dest, &close_src);
  auto total_nparens = AssignParenIds(*ofst, &paren_map);
  AssignParenLabels(total_nparens, parens);
  AddParensToFst(*parens, paren_map, open_dest, close_src,
                 close_non_term_weight, ofst);
  if (!fst_array.empty()) {
    ofst->SetInputSymbols(fst_array[0].second->InputSymbols());
    ofst->SetOutputSymbols(fst_array[0].second->OutputSymbols());
  }
  AddParensToSymbolTables(*parens, ofst);
}

template <class Arc>
size_t PdtLeftParser<Arc>::AssignParenIds(
    const Fst<Arc> &ofst,
    ParenMap *paren_map) const {
  // Number of distinct parenthesis pairs per FST.
  std::vector<size_t> nparens(FstArray().size(), 0);
  // Number of distinct parenthesis pairs overall.
  size_t total_nparens = 0;
  for (StateIterator<Fst<Arc>> siter(ofst); !siter.Done(); siter.Next()) {
    const auto os = siter.Value();
    for (ArcIterator<Fst<Arc>> aiter(ofst, os); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto nfst_id = Label2Id(arc.olabel);
      if (nfst_id != kNoStateId) {
        const ParenKey paren_key(nfst_id, arc.nextstate);
        auto it = paren_map->find(paren_key);
        if (it == paren_map->end()) {
          // Assigns new paren ID for this (FST, dest state) pair.
          (*paren_map)[paren_key] = nparens[nfst_id]++;
          if (nparens[nfst_id] > total_nparens)
            total_nparens = nparens[nfst_id];
        }
      }
    }
  }
  return total_nparens;
}

// Similar to PdtLeftParser but:
//
// 1. Uses epsilons rather than parentheses labels for any non-terminal
//    instances within a left- (right-) linear dependency SCC,
// 2. Allocates a paren ID uniquely for each such dependency SCC (rather than
//    non-terminal = dependency state) and destination state.
template <class Arc>
class PdtLeftSRParser final : public PdtParser<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using LabelFstPair = typename PdtParser<Arc>::LabelFstPair;
  using LabelPair = typename PdtParser<Arc>::LabelPair;
  using LabelStatePair = typename PdtParser<Arc>::LabelStatePair;
  using StateWeightPair = typename PdtParser<Arc>::StateWeightPair;
  using ParenKey = typename PdtParser<Arc>::ParenKey;
  using ParenMap = typename PdtParser<Arc>::ParenMap;

  using PdtParser<Arc>::AddParensToFst;
  using PdtParser<Arc>::AddParensToSymbolTables;
  using PdtParser<Arc>::AssignParenLabels;
  using PdtParser<Arc>::CreateFst;
  using PdtParser<Arc>::FstArray;
  using PdtParser<Arc>::GetLabelStatePair;
  using PdtParser<Arc>::GetState;
  using PdtParser<Arc>::Label2Id;
  using PdtParser<Arc>::Root;

  PdtLeftSRParser(const std::vector<LabelFstPair> &fst_array,
                  const PdtReplaceOptions<Arc> &opts) :
      PdtParser<Arc>(fst_array, opts),
      replace_util_(fst_array, ReplaceUtilOptions(opts.root)) { }

  void GetParser(MutableFst<Arc> *ofst,
                 std::vector<LabelPair> *parens) override;

 protected:
  // Assigns a unique parenthesis ID for each non-terminal, destination state
  // pair when the non-terminal refers to a non-linear FST. Otherwise, assigns
  // a unique parenthesis ID for each dependency SCC, destination state pair if
  // the non-terminal instance is between
  // SCCs. Otherwise does nothing.
  size_t AssignParenIds(const Fst<Arc> &ofst,
                        ParenMap *paren_map) const override;

  // Returns dependency SCC for given label.
  size_t SCC(Label label) const { return replace_util_.SCC(label); }

  // Is a given dependency SCC left-linear?
  bool SCCLeftLinear(size_t scc_id) const {
    const auto ll_props = kReplaceSCCLeftLinear | kReplaceSCCNonTrivial;
    const auto scc_props = replace_util_.SCCProperties(scc_id);
    return (scc_props & ll_props) == ll_props;
  }

  // Is a given dependency SCC right-linear?
  bool SCCRightLinear(size_t scc_id) const {
    const auto lr_props = kReplaceSCCRightLinear | kReplaceSCCNonTrivial;
    const auto scc_props = replace_util_.SCCProperties(scc_id);
    return (scc_props & lr_props) == lr_props;
  }

  // Components of left- (right-) linear dependency SCC; empty o.w.
  const std::vector<size_t> &SCCComps(size_t scc_id) const {
    if (scc_comps_.empty()) GetSCCComps();
    return scc_comps_[scc_id];
  }

  // Returns the representative state of an SCC. For left-linear grammars, it
  // is one of the initial states. For right-linear grammars, it is one of the
  // non-terminal destination states; otherwise, it is kNoStateId.
  StateId RepState(size_t scc_id) const {
    if (SCCComps(scc_id).empty()) return kNoStateId;
    const auto fst_id = SCCComps(scc_id).front();
    const auto &fst_array = FstArray();
    const auto label = fst_array[fst_id].first;
    const auto *ifst = fst_array[fst_id].second;
    if (SCCLeftLinear(scc_id)) {
      const LabelStatePair lsp(label, ifst->Start());
      return GetState(lsp);
    } else {  // Right-linear.
      const LabelStatePair lsp(label, *NonTermDests(fst_id).begin());
      return GetState(lsp);
    }
    return kNoStateId;
  }

 private:
  // Merges initial (final) states of in a left- (right-) linear dependency SCC
  // after dealing with the non-terminal arc and final weights.
  void ProcSCCs(MutableFst<Arc> *ofst,
                std::vector<StateId> *open_dest,
                std::vector<std::vector<StateWeightPair>> *close_src,
                std::vector<bool> *close_non_term_weight) const;

  // Computes components of left- (right-) linear dependency SCC.
  void GetSCCComps() const {
    const std::vector<LabelFstPair> &fst_array = FstArray();
    for (size_t i = 0; i < fst_array.size(); ++i) {
      const auto label = fst_array[i].first;
      const auto scc_id = SCC(label);
      if (scc_comps_.size() <= scc_id) scc_comps_.resize(scc_id + 1);
      if (SCCLeftLinear(scc_id) || SCCRightLinear(scc_id)) {
        scc_comps_[scc_id].push_back(i);
      }
    }
  }

  const std::set<StateId> &NonTermDests(StateId fst_id) const {
    if (non_term_dests_.empty()) GetNonTermDests();
    return non_term_dests_[fst_id];
  }

  // Finds non-terminal destination states for right-linear FSTS, or does
  // nothing if not found.
  void GetNonTermDests() const;

  // Dependency SCC info.
  mutable ReplaceUtil<Arc> replace_util_;
  // Components of left- (right-) linear dependency SCCs, or empty otherwise.
  mutable std::vector<std::vector<size_t>> scc_comps_;
  // States that have non-terminals entering them for each (right-linear) FST.
  mutable std::vector<std::set<StateId>> non_term_dests_;
};

template <class Arc>
void PdtLeftSRParser<Arc>::GetParser(
    MutableFst<Arc> *ofst,
    std::vector<LabelPair> *parens) {
  ofst->DeleteStates();
  parens->clear();
  const auto &fst_array = FstArray();
  // Map that gives the paren ID for a (non-terminal, dest. state) pair.
  ParenMap paren_map;
  // Specifies the open parenthesis destination state for a given non-terminal.
  // The source is the non-terminal instance source state.
  std::vector<StateId> open_dest(fst_array.size(), kNoStateId);
  // Specifies close parenthesis source states and weights for a given
  // non-terminal. The destination is the non-terminal instance destination
  // state.
  std::vector<std::vector<StateWeightPair>> close_src(fst_array.size());
  // Specifies non-terminals for which the non-terminal arc weight should be
  // applied on the close parenthesis (multiplying the close_src weight above)
  // rather than on the open parenthesis.
  std::vector<bool> close_non_term_weight(fst_array.size(), false);
  CreateFst(ofst, &open_dest, &close_src);
  ProcSCCs(ofst, &open_dest, &close_src, &close_non_term_weight);
  const auto total_nparens = AssignParenIds(*ofst, &paren_map);
  AssignParenLabels(total_nparens, parens);
  AddParensToFst(*parens, paren_map, open_dest, close_src,
                 close_non_term_weight, ofst);
  if (!fst_array.empty()) {
    ofst->SetInputSymbols(fst_array[0].second->InputSymbols());
    ofst->SetOutputSymbols(fst_array[0].second->OutputSymbols());
  }
  AddParensToSymbolTables(*parens, ofst);
  Connect(ofst);
}

template <class Arc>
void PdtLeftSRParser<Arc>::ProcSCCs(
    MutableFst<Arc> *ofst,
    std::vector<StateId> *open_dest,
    std::vector<std::vector<StateWeightPair>> *close_src,
    std::vector<bool> *close_non_term_weight) const {
  const auto &fst_array = FstArray();
  for (StateIterator<Fst<Arc>> siter(*ofst); !siter.Done(); siter.Next()) {
    const auto os = siter.Value();
    const auto label = GetLabelStatePair(os).first;
    const auto is = GetLabelStatePair(os).second;
    const auto fst_id = Label2Id(label);
    const auto scc_id = SCC(label);
    const auto rs = RepState(scc_id);
    const auto *ifst = fst_array[fst_id].second;
    // SCC LEFT-LINEAR: puts non-terminal weights on close parentheses. Merges
    // initial states into SCC representative state and updates open_dest.
    if (SCCLeftLinear(scc_id)) {
      (*close_non_term_weight)[fst_id] = true;
      if (is == ifst->Start() && os != rs) {
        for (ArcIterator<Fst<Arc>> aiter(*ofst, os); !aiter.Done();
             aiter.Next()) {
          const auto &arc = aiter.Value();
          ofst->AddArc(rs, arc);
        }
        ofst->DeleteArcs(os);
        if (os == ofst->Start())
          ofst->SetStart(rs);
        (*open_dest)[fst_id] = rs;
      }
    }
    // SCC RIGHT-LINEAR: pushes back final weights onto non-terminals, if
    // possible, or adds weighted epsilons to the SCC representative state.
    // Merges final states into SCC representative state and updates close_src.
    if (SCCRightLinear(scc_id)) {
      for (MutableArcIterator<MutableFst<Arc>> aiter(ofst, os); !aiter.Done();
           aiter.Next()) {
        auto arc = aiter.Value();
        const auto idest = GetLabelStatePair(arc.nextstate).second;
        if (NonTermDests(fst_id).count(idest) > 0) {
          if (ofst->Final(arc.nextstate) != Weight::Zero()) {
            ofst->SetFinal(arc.nextstate, Weight::Zero());
            ofst->SetFinal(rs, Weight::One());
          }
          arc.weight = Times(arc.weight, ifst->Final(idest));
          arc.nextstate = rs;
          aiter.SetValue(arc);
        }
      }
      const auto final_weight = ifst->Final(is);
      if (final_weight != Weight::Zero() &&
          NonTermDests(fst_id).count(is) == 0) {
        ofst->AddArc(os, Arc(0, 0, final_weight, rs));
        if (ofst->Final(os) != Weight::Zero()) {
          ofst->SetFinal(os, Weight::Zero());
          ofst->SetFinal(rs, Weight::One());
        }
      }
      if (is == ifst->Start()) {
        (*close_src)[fst_id].clear();
        (*close_src)[fst_id].emplace_back(rs, Weight::One());
      }
    }
  }
}

template <class Arc>
void PdtLeftSRParser<Arc>::GetNonTermDests() const {
  const auto &fst_array = FstArray();
  non_term_dests_.resize(fst_array.size());
  for (size_t fst_id = 0; fst_id < fst_array.size(); ++fst_id) {
    const auto label = fst_array[fst_id].first;
    const auto scc_id = SCC(label);
    if (SCCRightLinear(scc_id)) {
      const auto *ifst = fst_array[fst_id].second;
      for (StateIterator<Fst<Arc>> siter(*ifst); !siter.Done(); siter.Next()) {
        const auto is = siter.Value();
        for (ArcIterator<Fst<Arc>> aiter(*ifst, is); !aiter.Done();
             aiter.Next()) {
          const auto &arc = aiter.Value();
          if (Label2Id(arc.olabel) != kNoStateId) {
            non_term_dests_[fst_id].insert(arc.nextstate);
          }
        }
      }
    }
  }
}

template <class Arc>
size_t PdtLeftSRParser<Arc>::AssignParenIds(
    const Fst<Arc> &ofst,
    ParenMap *paren_map) const {
  const auto &fst_array = FstArray();
  // Number of distinct parenthesis pairs per FST.
  std::vector<size_t> nparens(fst_array.size(), 0);
  // Number of distinct parenthesis pairs overall.
  size_t total_nparens = 0;
  for (StateIterator<Fst<Arc>> siter(ofst); !siter.Done(); siter.Next()) {
    const auto os = siter.Value();
    const auto label = GetLabelStatePair(os).first;
    const auto scc_id = SCC(label);
    for (ArcIterator<Fst<Arc>> aiter(ofst, os); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      const auto nfst_id = Label2Id(arc.olabel);
      if (nfst_id != kNoStateId) {
        size_t nscc_id = SCC(arc.olabel);
        bool nscc_linear = !SCCComps(nscc_id).empty();
        // Assigns a parenthesis ID for the non-terminal transition
        // if the non-terminal belongs to a (left-/right-) linear dependency
        // SCC or if the transition is in an FST from a different SCC
        if (!nscc_linear || scc_id != nscc_id) {
          // For (left-/right-) linear SCCs instead of using nfst_id, we
          // will use its SCC prototype pfst_id for assigning distinct
          // parenthesis IDs.
          const auto pfst_id =
              nscc_linear ? SCCComps(nscc_id).front() : nfst_id;
          ParenKey paren_key(pfst_id, arc.nextstate);
          const auto it = paren_map->find(paren_key);
          if (it == paren_map->end()) {
            // Assigns new paren ID for this (FST/SCC, dest. state) pair.
            if (nscc_linear) {
              // This is mapping we'll need, but we also store (harmlessly)
              // for the prototype below so we can easily keep count per SCC.
              const ParenKey nparen_key(nfst_id, arc.nextstate);
              (*paren_map)[nparen_key] = nparens[pfst_id];
            }
            (*paren_map)[paren_key] = nparens[pfst_id]++;
            if (nparens[pfst_id] > total_nparens) {
              total_nparens = nparens[pfst_id];
            }
          }
        }
      }
    }
  }
  return total_nparens;
}

// Builds a pushdown transducer (PDT) from an RTN specification. The result is
// a PDT written to a mutable FST where some transitions are labeled with
// open or close parentheses. To be interpreted as a PDT, the parens must
// balance on a path (see PdtExpand()). The open/close parenthesis label pairs
// are returned in the parens argument.
template <class Arc>
void Replace(
    const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>
        &ifst_array,
    MutableFst<Arc> *ofst,
    std::vector<std::pair<typename Arc::Label, typename Arc::Label>> *parens,
    const PdtReplaceOptions<Arc> &opts) {
  switch (opts.type) {
    case PDT_LEFT_PARSER:
      {
        PdtLeftParser<Arc> pr(ifst_array, opts);
        pr.GetParser(ofst, parens);
        return;
      }
    case PDT_LEFT_SR_PARSER:
      {
        PdtLeftSRParser<Arc> pr(ifst_array, opts);
        pr.GetParser(ofst, parens);
        return;
      }
    default:
      FSTERROR() << "Replace: Unknown PDT parser type: " << opts.type;
      ofst->DeleteStates();
      ofst->SetProperties(kError, kError);
      parens->clear();
      return;
  }
}

// Variant where the only user-controlled arguments is the root ID.
template <class Arc>
void Replace(
    const std::vector<std::pair<typename Arc::Label, const Fst<Arc> *>>
        &ifst_array,
    MutableFst<Arc> *ofst,
    std::vector<std::pair<typename Arc::Label, typename Arc::Label>> *parens,
    typename Arc::Label root) {
  PdtReplaceOptions<Arc> opts(root);
  Replace(ifst_array, ofst, parens, opts);
}

}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_REPLACE_H_
