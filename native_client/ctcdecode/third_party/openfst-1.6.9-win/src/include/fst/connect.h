// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes and functions to remove unsuccessful paths from an FST.

#ifndef FST_CONNECT_H_
#define FST_CONNECT_H_

#include <vector>

#include <fst/dfs-visit.h>
#include <fst/mutable-fst.h>
#include <fst/union-find.h>


namespace fst {

// Finds and returns connected components. Use with Visit().
template <class Arc>
class CcVisitor {
 public:
  using Weight = typename Arc::Weight;
  using StateId = typename Arc::StateId;

  // cc[i]: connected component number for state i.
  explicit CcVisitor(std::vector<StateId> *cc)
      : comps_(new UnionFind<StateId>(0, kNoStateId)), cc_(cc), nstates_(0) {}

  // comps: connected components equiv classes.
  explicit CcVisitor(UnionFind<StateId> *comps)
      : comps_(comps), cc_(nullptr), nstates_(0) {}

  ~CcVisitor() {
    if (cc_) delete comps_;
  }

  void InitVisit(const Fst<Arc> &fst) {}

  bool InitState(StateId s, StateId root) {
    ++nstates_;
    if (comps_->FindSet(s) == kNoStateId) comps_->MakeSet(s);
    return true;
  }

  bool WhiteArc(StateId s, const Arc &arc) {
    comps_->MakeSet(arc.nextstate);
    comps_->Union(s, arc.nextstate);
    return true;
  }

  bool GreyArc(StateId s, const Arc &arc) {
    comps_->Union(s, arc.nextstate);
    return true;
  }

  bool BlackArc(StateId s, const Arc &arc) {
    comps_->Union(s, arc.nextstate);
    return true;
  }

  void FinishState(StateId s) {}

  void FinishVisit() {
    if (cc_) GetCcVector(cc_);
  }

  // Returns number of components.
  // cc[i]: connected component number for state i.
  int GetCcVector(std::vector<StateId> *cc) {
    cc->clear();
    cc->resize(nstates_, kNoStateId);
    StateId ncomp = 0;
    for (StateId s = 0; s < nstates_; ++s) {
      const auto rep = comps_->FindSet(s);
      auto &comp = (*cc)[rep];
      if (comp == kNoStateId) {
        comp = ncomp;
        ++ncomp;
      }
      (*cc)[s] = comp;
    }
    return ncomp;
  }

 private:
  UnionFind<StateId> *comps_;  // Components.
  std::vector<StateId> *cc_;   // State's cc number.
  StateId nstates_;            // State count.
};

// Finds and returns strongly-connected components, accessible and
// coaccessible states and related properties. Uses Tarjan's single
// DFS SCC algorithm (see Aho, et al, "Design and Analysis of Computer
// Algorithms", 189pp). Use with DfsVisit();
template <class Arc>
class SccVisitor {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // scc[i]: strongly-connected component number for state i.
  //   SCC numbers will be in topological order for acyclic input.
  // access[i]: accessibility of state i.
  // coaccess[i]: coaccessibility of state i.
  // Any of above can be NULL.
  // props: related property bits (cyclicity, initial cyclicity,
  //   accessibility, coaccessibility) set/cleared (o.w. unchanged).
  SccVisitor(std::vector<StateId> *scc, std::vector<bool> *access,
             std::vector<bool> *coaccess, uint64_t *props)
      : scc_(scc), access_(access), coaccess_(coaccess), props_(props) {}
  explicit SccVisitor(uint64_t *props)
      : scc_(nullptr), access_(nullptr), coaccess_(nullptr), props_(props) {}

  void InitVisit(const Fst<Arc> &fst);

  bool InitState(StateId s, StateId root);

  bool TreeArc(StateId s, const Arc &arc) { return true; }

  bool BackArc(StateId s, const Arc &arc) {
    const auto t = arc.nextstate;
    if ((*dfnumber_)[t] < (*lowlink_)[s]) (*lowlink_)[s] = (*dfnumber_)[t];
    if ((*coaccess_)[t]) (*coaccess_)[s] = true;
    *props_ |= kCyclic;
    *props_ &= ~kAcyclic;
    if (t == start_) {
      *props_ |= kInitialCyclic;
      *props_ &= ~kInitialAcyclic;
    }
    return true;
  }

  bool ForwardOrCrossArc(StateId s, const Arc &arc) {
    const auto t = arc.nextstate;
    if ((*dfnumber_)[t] < (*dfnumber_)[s] /* cross edge */ && (*onstack_)[t] &&
        (*dfnumber_)[t] < (*lowlink_)[s]) {
      (*lowlink_)[s] = (*dfnumber_)[t];
    }
    if ((*coaccess_)[t]) (*coaccess_)[s] = true;
    return true;
  }

  // Last argument always ignored, but required by the interface.
  void FinishState(StateId state, StateId p, const Arc *);

  void FinishVisit() {
    // Numbers SCCs in topological order when acyclic.
    if (scc_) {
      for (StateId s = 0; s < scc_->size(); ++s) {
        (*scc_)[s] = nscc_ - 1 - (*scc_)[s];
      }
    }
    if (coaccess_internal_) delete coaccess_;
    dfnumber_.reset();
    lowlink_.reset();
    onstack_.reset();
    scc_stack_.reset();
  }

 private:
  std::vector<StateId> *scc_;    // State's scc number.
  std::vector<bool> *access_;    // State's accessibility.
  std::vector<bool> *coaccess_;  // State's coaccessibility.
  uint64_t *props_;
  const Fst<Arc> *fst_;
  StateId start_;
  StateId nstates_;  // State count.
  StateId nscc_;     // SCC count.
  bool coaccess_internal_;
  std::unique_ptr<std::vector<StateId>> dfnumber_;  // State discovery times.
  std::unique_ptr<std::vector<StateId>>
      lowlink_;  // lowlink[state] == dfnumber[state] => SCC root
  std::unique_ptr<std::vector<bool>> onstack_;  // Is a state on the SCC stack?
  std::unique_ptr<std::vector<StateId>>
      scc_stack_;  // SCC stack, with random access.
};

template <class Arc>
inline void SccVisitor<Arc>::InitVisit(const Fst<Arc> &fst) {
  if (scc_) scc_->clear();
  if (access_) access_->clear();
  if (coaccess_) {
    coaccess_->clear();
    coaccess_internal_ = false;
  } else {
    coaccess_ = new std::vector<bool>;
    coaccess_internal_ = true;
  }
  *props_ |= kAcyclic | kInitialAcyclic | kAccessible | kCoAccessible;
  *props_ &= ~(kCyclic | kInitialCyclic | kNotAccessible | kNotCoAccessible);
  fst_ = &fst;
  start_ = fst.Start();
  nstates_ = 0;
  nscc_ = 0;
  dfnumber_.reset(new std::vector<StateId>());
  lowlink_.reset(new std::vector<StateId>());
  onstack_.reset(new std::vector<bool>());
  scc_stack_.reset(new std::vector<StateId>());
}

template <class Arc>
inline bool SccVisitor<Arc>::InitState(StateId s, StateId root) {
  scc_stack_->push_back(s);
  while (dfnumber_->size() <= s) {
    if (scc_) scc_->push_back(-1);
    if (access_) access_->push_back(false);
    coaccess_->push_back(false);
    dfnumber_->push_back(-1);
    lowlink_->push_back(-1);
    onstack_->push_back(false);
  }
  (*dfnumber_)[s] = nstates_;
  (*lowlink_)[s] = nstates_;
  (*onstack_)[s] = true;
  if (root == start_) {
    if (access_) (*access_)[s] = true;
  } else {
    if (access_) (*access_)[s] = false;
    *props_ |= kNotAccessible;
    *props_ &= ~kAccessible;
  }
  ++nstates_;
  return true;
}

template <class Arc>
inline void SccVisitor<Arc>::FinishState(StateId s, StateId p, const Arc *) {
  if (fst_->Final(s) != Weight::Zero()) (*coaccess_)[s] = true;
  if ((*dfnumber_)[s] == (*lowlink_)[s]) {  // Root of new SCC.
    bool scc_coaccess = false;
    auto i = scc_stack_->size();
    StateId t;
    do {
      t = (*scc_stack_)[--i];
      if ((*coaccess_)[t]) scc_coaccess = true;
    } while (s != t);
    do {
      t = scc_stack_->back();
      if (scc_) (*scc_)[t] = nscc_;
      if (scc_coaccess) (*coaccess_)[t] = true;
      (*onstack_)[t] = false;
      scc_stack_->pop_back();
    } while (s != t);
    if (!scc_coaccess) {
      *props_ |= kNotCoAccessible;
      *props_ &= ~kCoAccessible;
    }
    ++nscc_;
  }
  if (p != kNoStateId) {
    if ((*coaccess_)[s]) (*coaccess_)[p] = true;
    if ((*lowlink_)[s] < (*lowlink_)[p]) (*lowlink_)[p] = (*lowlink_)[s];
  }
}

// Trims an FST, removing states and arcs that are not on successful paths.
// This version modifies its input.
//
// Complexity:
//
//   Time:  O(V + E)
//   Space: O(V + E)
//
// where V = # of states and E = # of arcs.
template <class Arc>
void Connect(MutableFst<Arc> *fst) {
  using StateId = typename Arc::StateId;
  std::vector<bool> access;
  std::vector<bool> coaccess;
  uint64_t props = 0;
  SccVisitor<Arc> scc_visitor(nullptr, &access, &coaccess, &props);
  DfsVisit(*fst, &scc_visitor);
  std::vector<StateId> dstates;
  for (StateId s = 0; s < access.size(); ++s) {
    if (!access[s] || !coaccess[s]) dstates.push_back(s);
  }
  fst->DeleteStates(dstates);
  fst->SetProperties(kAccessible | kCoAccessible, kAccessible | kCoAccessible);
}

// Returns an acyclic FST where each SCC in the input FST has been condensed to
// a single state with transitions between SCCs retained and within SCCs
// dropped. Also populates 'scc' with a mapping from input to output states.
template <class Arc>
void Condense(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
              std::vector<typename Arc::StateId> *scc) {
  using StateId = typename Arc::StateId;
  ofst->DeleteStates();
  uint64_t props = 0;
  SccVisitor<Arc> scc_visitor(scc, nullptr, nullptr, &props);
  DfsVisit(ifst, &scc_visitor);
  for (StateId s = 0; s < scc->size(); ++s) {
    const auto c = (*scc)[s];
    while (c >= ofst->NumStates()) ofst->AddState();
    if (s == ifst.Start()) ofst->SetStart(c);
    const auto weight = ifst.Final(s);
    if (weight != Arc::Weight::Zero())
      ofst->SetFinal(c, Plus(ofst->Final(c), weight));
    for (ArcIterator<Fst<Arc>> aiter(ifst, s); !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      const auto nextc = (*scc)[arc.nextstate];
      if (nextc != c) {
        while (nextc >= ofst->NumStates()) ofst->AddState();
        arc.nextstate = nextc;
        ofst->AddArc(c, arc);
      }
    }
  }
  ofst->SetProperties(kAcyclic | kInitialAcyclic, kAcyclic | kInitialAcyclic);
}

}  // namespace fst

#endif  // FST_CONNECT_H_
