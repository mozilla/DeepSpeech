// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Queue-dependent visitation of finite-state transducers. See also dfs-visit.h.

#ifndef FST_VISIT_H_
#define FST_VISIT_H_


#include <fst/arcfilter.h>
#include <fst/mutable-fst.h>


namespace fst {

// Visitor Interface: class determining actions taken during a visit. If any of
// the boolean member functions return false, the visit is aborted by first
// calling FinishState() on all unfinished (grey) states and then calling
// FinishVisit().
//
// Note this is more general than the visitor interface in dfs-visit.h but lacks
// some DFS-specific behavior.
//
// template <class Arc>
// class Visitor {
//  public:
//   using StateId = typename Arc::StateId;
//
//   Visitor(T *return_data);
//
//   // Invoked before visit.
//   void InitVisit(const Fst<Arc> &fst);
//
//   // Invoked when state discovered (2nd arg is visitation root).
//   bool InitState(StateId s, StateId root);
//
//   // Invoked when arc to white/undiscovered state examined.
//   bool WhiteArc(StateId s, const Arc &arc);
//
//   // Invoked when arc to grey/unfinished state examined.
//   bool GreyArc(StateId s, const Arc &arc);
//
//   // Invoked when arc to black/finished state examined.
//   bool BlackArc(StateId s, const Arc &arc);
//
//   // Invoked when state finished.
//   void FinishState(StateId s);
//
//   // Invoked after visit.
//   void FinishVisit();
// };

// Performs queue-dependent visitation. Visitor class argument determines
// actions and contains any return data. ArcFilter determines arcs that are
// considered. If 'access_only' is true, performs visitation only to states
// accessible from the initial state.
template <class FST, class Visitor, class Queue, class ArcFilter>
void Visit(const FST &fst, Visitor *visitor, Queue *queue, ArcFilter filter,
           bool access_only = false) {
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;
  visitor->InitVisit(fst);
  const auto start = fst.Start();
  if (start == kNoStateId) {
    visitor->FinishVisit();
    return;
  }
  // An FST's state's visit color.
  static constexpr uint8 kWhiteState = 0x01;  // Undiscovered.
  static constexpr uint8 kGreyState = 0x02;   // Discovered & unfinished.
  static constexpr uint8 kBlackState = 0x04;  // Finished.
  // We destroy an iterator as soon as possible and mark it so.
  static constexpr uint8 kArcIterDone = 0x08;
  std::vector<uint8> state_status;
  std::vector<ArcIterator<FST> *> arc_iterator;
  MemoryPool<ArcIterator<FST>> aiter_pool;
  StateId nstates = start + 1;  // Number of known states in general case.
  bool expanded = false;
  if (fst.Properties(kExpanded, false)) {  // Tests if expanded, then uses
    nstates = CountStates(fst);            // ExpandedFst::NumStates().
    expanded = true;
  }
  state_status.resize(nstates, kWhiteState);
  arc_iterator.resize(nstates);
  StateIterator<Fst<Arc>> siter(fst);
  // Continues visit while true.
  bool visit = true;
  // Iterates over trees in visit forest.
  for (auto root = start; visit && root < nstates;) {
    visit = visitor->InitState(root, root);
    state_status[root] = kGreyState;
    queue->Enqueue(root);
    while (!queue->Empty()) {
      auto state = queue->Head();
      if (state >= state_status.size()) {
        nstates = state + 1;
        state_status.resize(nstates, kWhiteState);
        arc_iterator.resize(nstates);
      }
      // Creates arc iterator if needed.
      if (!arc_iterator[state] && !(state_status[state] & kArcIterDone) &&
          visit) {
        arc_iterator[state] = new (&aiter_pool) ArcIterator<FST>(fst, state);
      }
      // Deletes arc iterator if done.
      auto *aiter = arc_iterator[state];
      if ((aiter && aiter->Done()) || !visit) {
        Destroy(aiter, &aiter_pool);
        arc_iterator[state] = nullptr;
        state_status[state] |= kArcIterDone;
      }
      // Dequeues state and marks black if done.
      if (state_status[state] & kArcIterDone) {
        queue->Dequeue();
        visitor->FinishState(state);
        state_status[state] = kBlackState;
        continue;
      }
      const auto &arc = aiter->Value();
      if (arc.nextstate >= state_status.size()) {
        nstates = arc.nextstate + 1;
        state_status.resize(nstates, kWhiteState);
        arc_iterator.resize(nstates);
      }
      // Visits respective arc types.
      if (filter(arc)) {
        // Enqueues destination state and marks grey if white.
        if (state_status[arc.nextstate] == kWhiteState) {
          visit = visitor->WhiteArc(state, arc);
          if (!visit) continue;
          visit = visitor->InitState(arc.nextstate, root);
          state_status[arc.nextstate] = kGreyState;
          queue->Enqueue(arc.nextstate);
        } else if (state_status[arc.nextstate] == kBlackState) {
          visit = visitor->BlackArc(state, arc);
        } else {
          visit = visitor->GreyArc(state, arc);
        }
      }
      aiter->Next();
      // Destroys an iterator ASAP for efficiency.
      if (aiter->Done()) {
        Destroy(aiter, &aiter_pool);
        arc_iterator[state] = nullptr;
        state_status[state] |= kArcIterDone;
      }
    }
    if (access_only) break;
    // Finds next tree root.
    for (root = (root == start) ? 0 : root + 1;
         root < nstates && state_status[root] != kWhiteState; ++root) {
    }
    // Check for a state beyond the largest known state.
    if (!expanded && root == nstates) {
      for (; !siter.Done(); siter.Next()) {
        if (siter.Value() == nstates) {
          ++nstates;
          state_status.push_back(kWhiteState);
          arc_iterator.push_back(nullptr);
          break;
        }
      }
    }
  }
  visitor->FinishVisit();
}

template <class Arc, class Visitor, class Queue>
inline void Visit(const Fst<Arc> &fst, Visitor *visitor, Queue *queue) {
  Visit(fst, visitor, queue, AnyArcFilter<Arc>());
}

// Copies input FST to mutable FST following queue order.
template <class A>
class CopyVisitor {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  explicit CopyVisitor(MutableFst<Arc> *ofst) : ifst_(nullptr), ofst_(ofst) {}

  void InitVisit(const Fst<A> &ifst) {
    ifst_ = &ifst;
    ofst_->DeleteStates();
    ofst_->SetStart(ifst_->Start());
  }

  bool InitState(StateId state, StateId) {
    while (ofst_->NumStates() <= state) ofst_->AddState();
    return true;
  }

  bool WhiteArc(StateId state, const Arc &arc) {
    ofst_->AddArc(state, arc);
    return true;
  }

  bool GreyArc(StateId state, const Arc &arc) {
    ofst_->AddArc(state, arc);
    return true;
  }

  bool BlackArc(StateId state, const Arc &arc) {
    ofst_->AddArc(state, arc);
    return true;
  }

  void FinishState(StateId state) {
    ofst_->SetFinal(state, ifst_->Final(state));
  }

  void FinishVisit() {}

 private:
  const Fst<Arc> *ifst_;
  MutableFst<Arc> *ofst_;
};

// Visits input FST up to a state limit following queue order.
template <class A>
class PartialVisitor {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  explicit PartialVisitor(StateId maxvisit)
      : fst_(nullptr), maxvisit_(maxvisit) {}

  void InitVisit(const Fst<A> &ifst) {
    fst_ = &ifst;
    ninit_ = 0;
    nfinish_ = 0;
  }

  bool InitState(StateId state, StateId root) {
    ++ninit_;
    return ninit_ <= maxvisit_;
  }

  bool WhiteArc(StateId state, const Arc &arc) { return true; }

  bool GreyArc(StateId state, const Arc &arc) { return true; }

  bool BlackArc(StateId state, const Arc &arc) { return true; }

  void FinishState(StateId state) {
    fst_->Final(state);  // Visits super-final arc.
    ++nfinish_;
  }

  void FinishVisit() {}

  StateId NumInitialized() { return ninit_; }

  StateId NumFinished() { return nfinish_; }

 private:
  const Fst<Arc> *fst_;
  StateId maxvisit_;
  StateId ninit_;
  StateId nfinish_;
};

// Copies input FST to mutable FST up to a state limit following queue order.
template <class A>
class PartialCopyVisitor : public CopyVisitor<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using CopyVisitor<A>::WhiteArc;

  PartialCopyVisitor(MutableFst<Arc> *ofst, StateId maxvisit,
                     bool copy_grey = true, bool copy_black = true)
      : CopyVisitor<A>(ofst), maxvisit_(maxvisit),
        copy_grey_(copy_grey), copy_black_(copy_black) {}

  void InitVisit(const Fst<A> &ifst) {
    CopyVisitor<A>::InitVisit(ifst);
    ninit_ = 0;
    nfinish_ = 0;
  }

  bool InitState(StateId state, StateId root) {
    CopyVisitor<A>::InitState(state, root);
    ++ninit_;
    return ninit_ <= maxvisit_;
  }

  bool GreyArc(StateId state, const Arc &arc) {
    if (copy_grey_) return CopyVisitor<A>::GreyArc(state, arc);
    return true;
  }

  bool BlackArc(StateId state, const Arc &arc) {
    if (copy_black_) return CopyVisitor<A>::BlackArc(state, arc);
    return true;
  }

  void FinishState(StateId state) {
    CopyVisitor<A>::FinishState(state);
    ++nfinish_;
  }

  void FinishVisit() {}

  StateId NumInitialized() { return ninit_; }

  StateId NumFinished() { return nfinish_; }

 private:
  StateId maxvisit_;
  StateId ninit_;
  StateId nfinish_;
  const bool copy_grey_;
  const bool copy_black_;
};

}  // namespace fst

#endif  // FST_VISIT_H_
