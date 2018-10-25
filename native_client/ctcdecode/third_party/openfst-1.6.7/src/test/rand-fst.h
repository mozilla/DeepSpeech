#ifndef FST_TEST_RAND_FST_H_
#define FST_TEST_RAND_FST_H_

#include <fst/log.h>
#include <fst/mutable-fst.h>
#include <fst/verify.h>

namespace fst {

// Generates a random FST.
template <class Arc, class WeightGenerator>
void RandFst(const int num_random_states, const int num_random_arcs,
             const int num_random_labels, const float acyclic_prob,
             WeightGenerator *weight_generator, MutableFst<Arc> *fst) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  // Determines direction of the arcs wrt state numbering. This way we
  // can force acyclicity when desired.
  enum ArcDirection {
    ANY_DIRECTION = 0,
    FORWARD_DIRECTION = 1,
    REVERSE_DIRECTION = 2,
    NUM_DIRECTIONS = 3
  };

  ArcDirection arc_direction = ANY_DIRECTION;
  if (rand() / (RAND_MAX + 1.0) < acyclic_prob)
    arc_direction = rand() % 2 ? FORWARD_DIRECTION : REVERSE_DIRECTION;

  fst->DeleteStates();
  StateId ns = rand() % num_random_states;

  if (ns == 0) return;
  for (StateId s = 0; s < ns; ++s) fst->AddState();

  StateId start = rand() % ns;
  fst->SetStart(start);

  size_t na = rand() % num_random_arcs;
  for (size_t n = 0; n < na; ++n) {
    StateId s = rand() % ns;
    Arc arc;
    arc.ilabel = rand() % num_random_labels;
    arc.olabel = rand() % num_random_labels;
    arc.weight = (*weight_generator)();
    arc.nextstate = rand() % ns;

    if ((arc_direction == FORWARD_DIRECTION ||
         arc_direction == REVERSE_DIRECTION) &&
        s == arc.nextstate) {
      continue;  // skips self-loops
    }

    if ((arc_direction == FORWARD_DIRECTION && s > arc.nextstate) ||
        (arc_direction == REVERSE_DIRECTION && s < arc.nextstate)) {
      StateId t = s;  // reverses arcs
      s = arc.nextstate;
      arc.nextstate = t;
    }

    fst->AddArc(s, arc);
  }

  StateId nf = rand() % (ns + 1);
  for (StateId n = 0; n < nf; ++n) {
    StateId s = rand() % ns;
    Weight final = (*weight_generator)();
    fst->SetFinal(s, final);
  }
  VLOG(1) << "Check FST for sanity (including property bits).";
  CHECK(Verify(*fst));

  // Get/compute all properties.
  uint64 props = fst->Properties(kFstProperties, true);

  // Select random set of properties to be unknown.
  uint64 mask = 0;
  for (int n = 0; n < 8; ++n) {
    mask |= rand() & 0xff;
    mask <<= 8;
  }
  mask &= ~kTrinaryProperties;
  fst->SetProperties(props & ~mask, mask);
}

}  // namespace fst

#endif  // FST_TEST_RAND_FST_H_
