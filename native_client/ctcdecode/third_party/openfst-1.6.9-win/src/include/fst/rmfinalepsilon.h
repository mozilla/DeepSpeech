// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Function to remove of final states that have epsilon-only input arcs.

#ifndef FST_RMFINALEPSILON_H_
#define FST_RMFINALEPSILON_H_

#include <unordered_set>
#include <vector>

#include <fst/connect.h>
#include <fst/mutable-fst.h>


namespace fst {

// Removes final states that have epsilon-only input arcs.
template <class Arc>
void RmFinalEpsilon(MutableFst<Arc> *fst) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  // Determines the coaccesibility of states.
  std::vector<bool> access;
  std::vector<bool> coaccess;
  uint64_t props = 0;
  SccVisitor<Arc> scc_visitor(nullptr, &access, &coaccess, &props);
  DfsVisit(*fst, &scc_visitor);
  // Finds potential list of removable final states. These are final states that
  // have no outgoing transitions or final states that have a non-coaccessible
  // future.
  std::unordered_set<StateId> finals;
  for (StateIterator<Fst<Arc>> siter(*fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    if (fst->Final(s) != Weight::Zero()) {
      bool future_coaccess = false;
      for (ArcIterator<Fst<Arc>> aiter(*fst, s); !aiter.Done(); aiter.Next()) {
        const auto &arc = aiter.Value();
        if (coaccess[arc.nextstate]) {
          future_coaccess = true;
          break;
        }
      }
      if (!future_coaccess) finals.insert(s);
    }
  }
  // Moves the final weight.
  std::vector<Arc> arcs;
  for (StateIterator<Fst<Arc>> siter(*fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    auto weight = fst->Final(s);
    arcs.clear();
    for (ArcIterator<Fst<Arc>> aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      // Next state is in the list of finals.
      if (finals.find(arc.nextstate) != finals.end()) {
        // Sums up all epsilon arcs.
        if (arc.ilabel == 0 && arc.olabel == 0) {
          weight = Plus(Times(fst->Final(arc.nextstate), arc.weight), weight);
        } else {
          arcs.push_back(arc);
        }
      } else {
        arcs.push_back(arc);
      }
    }
    // If some arcs (epsilon arcs) were deleted, delete all arcs and add back
    // only the non-epsilon arcs.
    if (arcs.size() < fst->NumArcs(s)) {
      fst->DeleteArcs(s);
      fst->SetFinal(s, weight);
      for (const auto &arc : arcs) fst->AddArc(s, arc);
    }
  }
  Connect(fst);
}

}  // namespace fst

#endif  // FST_RMFINALEPSILON_H_
