// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions to manipulate and test property bits.

#ifndef FST_TEST_PROPERTIES_H_
#define FST_TEST_PROPERTIES_H_

#include <unordered_set>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/connect.h>
#include <fst/dfs-visit.h>


DECLARE_bool(fst_verify_properties);

namespace fst {
// namespace internal {

// For a binary property, the bit is always returned set. For a trinary (i.e.,
// two-bit) property, both bits are returned set iff either corresponding input
// bit is set.
inline uint64_t KnownProperties(uint64_t props) {
  return kBinaryProperties | (props & kTrinaryProperties) |
         ((props & kPosTrinaryProperties) << 1) |
         ((props & kNegTrinaryProperties) >> 1);
}

// Tests compatibility between two sets of properties.
inline bool CompatProperties(uint64_t props1, uint64_t props2) {
  const auto known_props1 = KnownProperties(props1);
  const auto known_props2 = KnownProperties(props2);
  const auto known_props = known_props1 & known_props2;
  const auto incompat_props = (props1 & known_props) ^ (props2 & known_props);
  if (incompat_props) {
    uint64_t prop = 1;
    for (int i = 0; i < 64; ++i, prop <<= 1) {
      if (prop & incompat_props) {
        LOG(ERROR) << "CompatProperties: Mismatch: " << PropertyNames[i]
                   << ": props1 = " << (props1 & prop ? "true" : "false")
                   << ", props2 = " << (props2 & prop ? "true" : "false");
      }
    }
    return false;
  } else {
    return true;
  }
}

// Computes FST property values defined in properties.h. The value of each
// property indicated in the mask will be determined and returned (these will
// never be unknown here). In the course of determining the properties
// specifically requested in the mask, certain other properties may be
// determined (those with little additional expense) and their values will be
// returned as well. The complete set of known properties (whether true or
// false) determined by this operation will be assigned to the the value pointed
// to by KNOWN. If 'use_stored' is true, pre-computed FST properties may be used
// when possible. 'mask & required_mask' is used to determine whether the stored
// propertoes can be used. This routine is seldom called directly; instead it is
// used to implement fst.Properties(mask, true).
template <class Arc>
uint64_t ComputeProperties(const Fst<Arc> &fst, uint64_t mask, uint64_t *known,
                         bool use_stored) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  const auto fst_props = fst.Properties(kFstProperties, false);  // FST-stored.
  // Check stored FST properties first if allowed.
  if (use_stored) {
    const auto known_props = KnownProperties(fst_props);
    // If FST contains required info, return it.
    if ((known_props & mask) == mask) {
      if (known) *known = known_props;
      return fst_props;
    }
  }
  // Computes (trinary) properties explicitly.
  // Initialize with binary properties (already known).
  uint64_t comp_props = fst_props & kBinaryProperties;
  // Computes these trinary properties with a DFS. We compute only those that
  // need a DFS here, since we otherwise would like to avoid a DFS since its
  // stack could grow large.
  uint64_t dfs_props = kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic |
                     kAccessible | kNotAccessible | kCoAccessible |
                     kNotCoAccessible;
  std::vector<StateId> scc;
  if (mask & (dfs_props | kWeightedCycles | kUnweightedCycles)) {
    SccVisitor<Arc> scc_visitor(&scc, nullptr, nullptr, &comp_props);
    DfsVisit(fst, &scc_visitor);
  }
  // Computes any remaining trinary properties via a state and arcs iterations
  if (mask & ~(kBinaryProperties | dfs_props)) {
    comp_props |= kAcceptor | kNoEpsilons | kNoIEpsilons | kNoOEpsilons |
                  kILabelSorted | kOLabelSorted | kUnweighted | kTopSorted |
                  kString;
    if (mask & (kIDeterministic | kNonIDeterministic)) {
      comp_props |= kIDeterministic;
    }
    if (mask & (kODeterministic | kNonODeterministic)) {
      comp_props |= kODeterministic;
    }
    if (mask & (dfs_props | kWeightedCycles | kUnweightedCycles)) {
      comp_props |= kUnweightedCycles;
    }
    std::unique_ptr<std::unordered_set<Label>> ilabels;
    std::unique_ptr<std::unordered_set<Label>> olabels;
    StateId nfinal = 0;
    for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
      StateId s = siter.Value();
      Arc prev_arc;
      // Creates these only if we need to.
      if (mask & (kIDeterministic | kNonIDeterministic)) {
        ilabels.reset(new std::unordered_set<Label>());
      }
      if (mask & (kODeterministic | kNonODeterministic)) {
        olabels.reset(new std::unordered_set<Label>());
      }
      bool first_arc = true;
      for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
        const auto &arc = aiter.Value();
        if (ilabels && ilabels->find(arc.ilabel) != ilabels->end()) {
          comp_props |= kNonIDeterministic;
          comp_props &= ~kIDeterministic;
        }
        if (olabels && olabels->find(arc.olabel) != olabels->end()) {
          comp_props |= kNonODeterministic;
          comp_props &= ~kODeterministic;
        }
        if (arc.ilabel != arc.olabel) {
          comp_props |= kNotAcceptor;
          comp_props &= ~kAcceptor;
        }
        if (arc.ilabel == 0 && arc.olabel == 0) {
          comp_props |= kEpsilons;
          comp_props &= ~kNoEpsilons;
        }
        if (arc.ilabel == 0) {
          comp_props |= kIEpsilons;
          comp_props &= ~kNoIEpsilons;
        }
        if (arc.olabel == 0) {
          comp_props |= kOEpsilons;
          comp_props &= ~kNoOEpsilons;
        }
        if (!first_arc) {
          if (arc.ilabel < prev_arc.ilabel) {
            comp_props |= kNotILabelSorted;
            comp_props &= ~kILabelSorted;
          }
          if (arc.olabel < prev_arc.olabel) {
            comp_props |= kNotOLabelSorted;
            comp_props &= ~kOLabelSorted;
          }
        }
        if (arc.weight != Weight::One() && arc.weight != Weight::Zero()) {
          comp_props |= kWeighted;
          comp_props &= ~kUnweighted;
          if ((comp_props & kUnweightedCycles) &&
              scc[s] == scc[arc.nextstate]) {
            comp_props |= kWeightedCycles;
            comp_props &= ~kUnweightedCycles;
          }
        }
        if (arc.nextstate <= s) {
          comp_props |= kNotTopSorted;
          comp_props &= ~kTopSorted;
        }
        if (arc.nextstate != s + 1) {
          comp_props |= kNotString;
          comp_props &= ~kString;
        }
        prev_arc = arc;
        first_arc = false;
        if (ilabels) ilabels->insert(arc.ilabel);
        if (olabels) olabels->insert(arc.olabel);
      }

      if (nfinal > 0) {  // Final state not last.
        comp_props |= kNotString;
        comp_props &= ~kString;
      }
      const auto final_weight = fst.Final(s);
      if (final_weight != Weight::Zero()) {  // Final state.
        if (final_weight != Weight::One()) {
          comp_props |= kWeighted;
          comp_props &= ~kUnweighted;
        }
        ++nfinal;
      } else {  // Non-final state.
        if (fst.NumArcs(s) != 1) {
          comp_props |= kNotString;
          comp_props &= ~kString;
        }
      }
    }
    if (fst.Start() != kNoStateId && fst.Start() != 0) {
      comp_props |= kNotString;
      comp_props &= ~kString;
    }
  }
  if (known) *known = KnownProperties(comp_props);
  return comp_props;
}

// This is a wrapper around ComputeProperties that will cause a fatal error if
// the stored properties and the computed properties are incompatible when
// FLAGS_fst_verify_properties is true. This routine is seldom called directly;
// instead it is used to implement fst.Properties(mask, true).
template <class Arc>
uint64_t TestProperties(const Fst<Arc> &fst, uint64_t mask, uint64_t *known) {
  if (FLAGS_fst_verify_properties) {
    const auto stored_props = fst.Properties(kFstProperties, false);
    const auto computed_props = ComputeProperties(fst, mask, known, false);
    if (!CompatProperties(stored_props, computed_props)) {
      FSTERROR() << "TestProperties: stored FST properties incorrect"
                 << " (stored: props1, computed: props2)";
    }
    return computed_props;
  } else {
    return ComputeProperties(fst, mask, known, true);
  }
}

// If all the properties of 'fst' corresponding to 'check_mask' are known,
// returns the stored properties. Otherwise, the properties corresponding to
// both 'check_mask' and 'test_mask' are computed. This is used to check for
// newly-added properties that might not be set in old binary files.
template <class Arc>
uint64_t CheckProperties(const Fst<Arc> &fst, uint64_t check_mask,
                       uint64_t test_mask) {
  auto props = fst.Properties(kFstProperties, false);
  if (FLAGS_fst_verify_properties) {
    props = TestProperties(fst, check_mask | test_mask, nullptr);
  } else if ((KnownProperties(props) & check_mask) != check_mask) {
    props = ComputeProperties(fst, check_mask | test_mask, nullptr, false);
  }
  return props & (check_mask | test_mask);
}

//}  // namespace internal
}  // namespace fst

#endif  // FST_TEST_PROPERTIES_H_
