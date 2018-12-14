// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions for updating property bits for various FST operations and
// string names of the properties.

#include <fst/properties.h>

#include <stddef.h>
#include <vector>

namespace fst {

// These functions determine the properties associated with the FST result of
// various finite-state operations. The property arguments correspond to the
// operation's FST arguments. The properties returned assume the operation
// modifies its first argument. Bitwise-and this result with kCopyProperties for
// the case when a new (possibly delayed) FST is instead constructed.

// Properties for a concatenatively-closed FST.
uint64_t ClosureProperties(uint64_t inprops, bool star, bool delayed) {
  auto outprops = (kError | kAcceptor | kUnweighted | kAccessible) & inprops;
  if (inprops & kUnweighted) outprops |= kUnweightedCycles;
  if (!delayed) {
    outprops |=
        (kExpanded | kMutable | kCoAccessible | kNotTopSorted | kNotString) &
        inprops;
  }
  if (!delayed || inprops & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kNotILabelSorted | kNotOLabelSorted | kWeighted |
                 kWeightedCycles | kNotAccessible | kNotCoAccessible) & inprops;
    if ((inprops & kWeighted) && (inprops & kAccessible) &&
        (inprops & kCoAccessible)) {
        outprops |= kWeightedCycles;
    }
  }
  return outprops;
}

// Properties for a complemented FST.
uint64_t ComplementProperties(uint64_t inprops) {
  auto outprops = kAcceptor | kUnweighted | kUnweightedCycles | kNoEpsilons |
                  kNoIEpsilons | kNoOEpsilons | kIDeterministic |
                  kODeterministic | kAccessible;
  outprops |=
      (kError | kILabelSorted | kOLabelSorted | kInitialCyclic) & inprops;
  if (inprops & kAccessible) {
    outprops |= kNotILabelSorted | kNotOLabelSorted | kCyclic;
  }
  return outprops;
}

// Properties for a composed FST.
uint64_t ComposeProperties(uint64_t inprops1, uint64_t inprops2) {
  auto outprops = kError & (inprops1 | inprops2);
  if (inprops1 & kAcceptor && inprops2 & kAcceptor) {
    outprops |= kAcceptor | kAccessible;
    outprops |= (kNoEpsilons | kNoIEpsilons | kNoOEpsilons | kAcyclic |
                 kInitialAcyclic) &
                inprops1 & inprops2;
    if (kNoIEpsilons & inprops1 & inprops2) {
      outprops |= (kIDeterministic | kODeterministic) & inprops1 & inprops2;
    }
  } else {
    outprops |= kAccessible;
    outprops |= (kAcceptor | kNoIEpsilons | kAcyclic | kInitialAcyclic) &
                inprops1 & inprops2;
    if (kNoIEpsilons & inprops1 & inprops2) {
      outprops |= kIDeterministic & inprops1 & inprops2;
    }
  }
  return outprops;
}

// Properties for a concatenated FST.
uint64_t ConcatProperties(uint64_t inprops1, uint64_t inprops2, bool delayed) {
  auto outprops = (kAcceptor | kUnweighted | kUnweightedCycles | kAcyclic) &
                  inprops1 & inprops2;
  outprops |= kError & (inprops1 | inprops2);
  const bool empty1 = delayed;  // Can the first FST be the empty machine?
  const bool empty2 = delayed;  // Can the second FST be the empty machine?
  if (!delayed) {
    outprops |= (kExpanded | kMutable | kNotTopSorted | kNotString) & inprops1;
    outprops |= (kNotTopSorted | kNotString) & inprops2;
  }
  if (!empty1) outprops |= (kInitialAcyclic | kInitialCyclic) & inprops1;
  if (!delayed || inprops1 & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kEpsilons | kIEpsilons | kOEpsilons | kNotILabelSorted |
                 kNotOLabelSorted | kWeighted | kWeightedCycles | kCyclic |
                 kNotAccessible | kNotCoAccessible) &
                inprops1;
  }
  if ((inprops1 & (kAccessible | kCoAccessible)) ==
          (kAccessible | kCoAccessible) &&
      !empty1) {
    outprops |= kAccessible & inprops2;
    if (!empty2) outprops |= kCoAccessible & inprops2;
    if (!delayed || inprops2 & kAccessible) {
      outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                   kEpsilons | kIEpsilons | kOEpsilons | kNotILabelSorted |
                   kNotOLabelSorted | kWeighted | kWeightedCycles | kCyclic |
                   kNotAccessible | kNotCoAccessible) &
                  inprops2;
    }
  }
  return outprops;
}

// Properties for a determinized FST.
uint64_t DeterminizeProperties(uint64_t inprops, bool has_subsequential_label,
                             bool distinct_psubsequential_labels) {
  auto outprops = kAccessible;
  if ((kAcceptor & inprops) ||
      ((kNoIEpsilons & inprops) && distinct_psubsequential_labels) ||
      (has_subsequential_label && distinct_psubsequential_labels)) {
    outprops |= kIDeterministic;
  }
  outprops |= (kError | kAcceptor | kAcyclic | kInitialAcyclic | kCoAccessible |
               kString) &
              inprops;
  if ((inprops & kNoIEpsilons) && distinct_psubsequential_labels) {
    outprops |= kNoEpsilons & inprops;
  }
  if (inprops & kAccessible) {
    outprops |= (kIEpsilons | kOEpsilons | kCyclic) & inprops;
  }
  if (inprops & kAcceptor) outprops |= (kNoIEpsilons | kNoOEpsilons) & inprops;
  if ((inprops & kNoIEpsilons) && has_subsequential_label) {
    outprops |= kNoIEpsilons;
  }
  return outprops;
}

// Properties for factored weight FST.
uint64_t FactorWeightProperties(uint64_t inprops) {
  auto outprops = (kExpanded | kMutable | kError | kAcceptor | kAcyclic |
                   kAccessible | kCoAccessible) &
                  inprops;
  if (inprops & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kEpsilons | kIEpsilons | kOEpsilons | kCyclic |
                 kNotILabelSorted | kNotOLabelSorted) &
                inprops;
  }
  return outprops;
}

// Properties for an inverted FST.
uint64_t InvertProperties(uint64_t inprops) {
  auto outprops = (kExpanded | kMutable | kError | kAcceptor | kNotAcceptor |
                   kEpsilons | kNoEpsilons | kWeighted | kUnweighted |
                   kWeightedCycles | kUnweightedCycles | kCyclic | kAcyclic |
                   kInitialCyclic | kInitialAcyclic | kTopSorted |
                   kNotTopSorted | kAccessible | kNotAccessible |
                   kCoAccessible | kNotCoAccessible | kString | kNotString) &
                  inprops;
  if (kIDeterministic & inprops) outprops |= kODeterministic;
  if (kNonIDeterministic & inprops) outprops |= kNonODeterministic;
  if (kODeterministic & inprops) outprops |= kIDeterministic;
  if (kNonODeterministic & inprops) outprops |= kNonIDeterministic;

  if (kIEpsilons & inprops) outprops |= kOEpsilons;
  if (kNoIEpsilons & inprops) outprops |= kNoOEpsilons;
  if (kOEpsilons & inprops) outprops |= kIEpsilons;
  if (kNoOEpsilons & inprops) outprops |= kNoIEpsilons;

  if (kILabelSorted & inprops) outprops |= kOLabelSorted;
  if (kNotILabelSorted & inprops) outprops |= kNotOLabelSorted;
  if (kOLabelSorted & inprops) outprops |= kILabelSorted;
  if (kNotOLabelSorted & inprops) outprops |= kNotILabelSorted;
  return outprops;
}

// Properties for a projected FST.
uint64_t ProjectProperties(uint64_t inprops, bool project_input) {
  auto outprops = kAcceptor;
  outprops |= (kExpanded | kMutable | kError | kWeighted | kUnweighted |
               kWeightedCycles | kUnweightedCycles |
               kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic |
               kTopSorted | kNotTopSorted | kAccessible | kNotAccessible |
               kCoAccessible | kNotCoAccessible | kString | kNotString) &
              inprops;
  if (project_input) {
    outprops |= (kIDeterministic | kNonIDeterministic | kIEpsilons |
                 kNoIEpsilons | kILabelSorted | kNotILabelSorted) &
                inprops;

    if (kIDeterministic & inprops) outprops |= kODeterministic;
    if (kNonIDeterministic & inprops) outprops |= kNonODeterministic;

    if (kIEpsilons & inprops) outprops |= kOEpsilons | kEpsilons;
    if (kNoIEpsilons & inprops) outprops |= kNoOEpsilons | kNoEpsilons;

    if (kILabelSorted & inprops) outprops |= kOLabelSorted;
    if (kNotILabelSorted & inprops) outprops |= kNotOLabelSorted;
  } else {
    outprops |= (kODeterministic | kNonODeterministic | kOEpsilons |
                 kNoOEpsilons | kOLabelSorted | kNotOLabelSorted) &
                inprops;

    if (kODeterministic & inprops) outprops |= kIDeterministic;
    if (kNonODeterministic & inprops) outprops |= kNonIDeterministic;

    if (kOEpsilons & inprops) outprops |= kIEpsilons | kEpsilons;
    if (kNoOEpsilons & inprops) outprops |= kNoIEpsilons | kNoEpsilons;

    if (kOLabelSorted & inprops) outprops |= kILabelSorted;
    if (kNotOLabelSorted & inprops) outprops |= kNotILabelSorted;
  }
  return outprops;
}

// Properties for a randgen FST.
uint64_t RandGenProperties(uint64_t inprops, bool weighted) {
  auto outprops = kAcyclic | kInitialAcyclic | kAccessible | kUnweightedCycles;
  outprops |= inprops & kError;
  if (weighted) {
    outprops |= kTopSorted;
    outprops |=
        (kAcceptor | kNoEpsilons | kNoIEpsilons | kNoOEpsilons |
         kIDeterministic | kODeterministic | kILabelSorted | kOLabelSorted) &
        inprops;
  } else {
    outprops |= kUnweighted;
    outprops |= (kAcceptor | kILabelSorted | kOLabelSorted) & inprops;
  }
  return outprops;
}

// Properties for a replace FST.
uint64_t ReplaceProperties(const std::vector<uint64_t>& inprops, std::ptrdiff_t root,
    bool epsilon_on_call, bool epsilon_on_return,
    bool out_epsilon_on_call, bool out_epsilon_on_return,
    bool replace_transducer, bool no_empty_fsts,
    bool all_ilabel_sorted, bool all_olabel_sorted,
    bool all_negative_or_dense) {
    if (inprops.empty()) return kNullProperties;
    uint64_t outprops = 0;
    for (auto inprop : inprops) outprops |= kError & inprop;
    uint64_t access_props = no_empty_fsts ? kAccessible | kCoAccessible : 0;
    for (auto inprop : inprops) {
        access_props &= (inprop & (kAccessible | kCoAccessible));
    }
    if (access_props == (kAccessible | kCoAccessible)) {
        outprops |= access_props;
        if (inprops[root] & kInitialCyclic) outprops |= kInitialCyclic;
        uint64_t props = 0;
        bool string = true;
        for (auto inprop : inprops) {
            if (replace_transducer) props |= kNotAcceptor & inprop;
            props |= (kNonIDeterministic | kNonODeterministic | kEpsilons |
                kIEpsilons | kOEpsilons | kWeighted | kWeightedCycles |
                kCyclic | kNotTopSorted | kNotString) & inprop;
            if (!(inprop & kString)) string = false;
        }
        outprops |= props;
        if (string) outprops |= kString;
    }
    bool acceptor = !replace_transducer;
    bool ideterministic = !epsilon_on_call && epsilon_on_return;
    bool no_iepsilons = !epsilon_on_call && !epsilon_on_return;
    bool acyclic = true;
    bool unweighted = true;
    for (size_t i = 0; i < inprops.size(); ++i) {
        if (!(inprops[i] & kAcceptor)) acceptor = false;
        if (!(inprops[i] & kIDeterministic)) ideterministic = false;
        if (!(inprops[i] & kNoIEpsilons)) no_iepsilons = false;
        if (!(inprops[i] & kAcyclic)) acyclic = false;
        if (!(inprops[i] & kUnweighted)) unweighted = false;
        if (i != root && !(inprops[i] & kNoIEpsilons)) ideterministic = false;
    }
    if (acceptor) outprops |= kAcceptor;
    if (ideterministic) outprops |= kIDeterministic;
    if (no_iepsilons) outprops |= kNoIEpsilons;
    if (acyclic) outprops |= kAcyclic;
    if (unweighted) outprops |= kUnweighted;
    if (inprops[root] & kInitialAcyclic) outprops |= kInitialAcyclic;
    // We assume that all terminals are positive. The resulting ReplaceFst is
    // known to be kILabelSorted when: (1) all sub-FSTs are kILabelSorted, (2) the
    // input label of the return arc is epsilon, and (3) one of the 3 following
    // conditions is satisfied:
    //
    //  1. the input label of the call arc is not epsilon
    //  2. all non-terminals are negative, or
    //  3. all non-terninals are positive and form a dense range containing 1.
    if (all_ilabel_sorted && epsilon_on_return &&
        (!epsilon_on_call || all_negative_or_dense)) {
        outprops |= kILabelSorted;
    }
    // Similarly, the resulting ReplaceFst is known to be kOLabelSorted when: (1)
    // all sub-FSTs are kOLabelSorted, (2) the output label of the return arc is
    // epsilon, and (3) one of the 3 following conditions is satisfied:
    //
    //  1. the output label of the call arc is not epsilon
    //  2. all non-terminals are negative, or
    //  3. all non-terninals are positive and form a dense range containing 1.
    if (all_olabel_sorted && out_epsilon_on_return &&
        (!out_epsilon_on_call || all_negative_or_dense)) {
        outprops |= kOLabelSorted;
    }
    return outprops;
}

// Properties for a relabeled FST.
uint64_t RelabelProperties(uint64_t inprops) {
  static constexpr auto outprops =
      kExpanded | kMutable | kError | kWeighted | kUnweighted |
      kWeightedCycles | kUnweightedCycles | kCyclic | kAcyclic |
      kInitialCyclic | kInitialAcyclic | kTopSorted | kNotTopSorted |
      kAccessible | kNotAccessible | kCoAccessible | kNotCoAccessible |
      kString | kNotString;
  return outprops & inprops;
}

// Properties for a reversed FST (the superinitial state limits this set).
uint64_t ReverseProperties(uint64_t inprops, bool has_superinitial) {
  auto outprops = (kExpanded | kMutable | kError | kAcceptor | kNotAcceptor |
                   kEpsilons | kIEpsilons | kOEpsilons | kUnweighted | kCyclic |
                   kAcyclic | kWeightedCycles | kUnweightedCycles) &
                  inprops;
  if (has_superinitial) outprops |= kWeighted & inprops;
  return outprops;
}

// Properties for re-weighted FST.
uint64_t ReweightProperties(uint64_t inprops) {
  auto outprops = inprops & kWeightInvariantProperties;
  outprops = outprops & ~kCoAccessible;
  return outprops;
}

// Properties for an epsilon-removed FST.
uint64_t RmEpsilonProperties(uint64_t inprops, bool delayed) {
  auto outprops = kNoEpsilons;
  outprops |= (kError | kAcceptor | kAcyclic | kInitialAcyclic) & inprops;
  if (inprops & kAcceptor) outprops |= kNoIEpsilons | kNoOEpsilons;
  if (!delayed) {
    outprops |= kExpanded | kMutable;
    outprops |= kTopSorted & inprops;
  }
  if (!delayed || inprops & kAccessible) outprops |= kNotAcceptor & inprops;
  return outprops;
}

// Properties for shortest path. This function computes how the properties of
// the output of shortest path need to be updated, given that 'props' is already
// known.
uint64_t ShortestPathProperties(uint64_t props, bool tree) {
  auto outprops =
      props | kAcyclic | kInitialAcyclic | kAccessible | kUnweightedCycles;
  if (!tree) outprops |= kCoAccessible;
  return outprops;
}

// Properties for a synchronized FST.
uint64_t SynchronizeProperties(uint64_t inprops) {
  auto outprops = (kError | kAcceptor | kAcyclic | kAccessible | kCoAccessible |
                   kUnweighted | kUnweightedCycles) &
                  inprops;
  if (inprops & kAccessible) {
    outprops |= (kCyclic | kNotCoAccessible | kWeighted | kWeightedCycles) &
        inprops;
  }
  return outprops;
}

// Properties for a unioned FST.
uint64_t UnionProperties(uint64_t inprops1, uint64_t inprops2, bool delayed) {
  auto outprops =
      (kAcceptor | kUnweighted | kUnweightedCycles | kAcyclic | kAccessible) &
      inprops1 & inprops2;
  outprops |= kError & (inprops1 | inprops2);
  outprops |= kInitialAcyclic;
  bool empty1 = delayed;  // Can the first FST be the empty machine?
  bool empty2 = delayed;  // Can the second FST be the empty machine?
  if (!delayed) {
    outprops |= (kExpanded | kMutable | kNotTopSorted) & inprops1;
    outprops |= kNotTopSorted & inprops2;
  }
  if (!empty1 && !empty2) {
    outprops |= kEpsilons | kIEpsilons | kOEpsilons;
    outprops |= kCoAccessible & inprops1 & inprops2;
  }
  // Note kNotCoAccessible does not hold because of kInitialAcyclic option.
  if (!delayed || inprops1 & kAccessible) {
    outprops |=
        (kNotAcceptor | kNonIDeterministic | kNonODeterministic | kEpsilons |
         kIEpsilons | kOEpsilons | kNotILabelSorted | kNotOLabelSorted |
         kWeighted | kWeightedCycles | kCyclic | kNotAccessible) &
        inprops1;
  }
  if (!delayed || inprops2 & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kEpsilons | kIEpsilons | kOEpsilons | kNotILabelSorted |
                 kNotOLabelSorted | kWeighted | kWeightedCycles | kCyclic |
                 kNotAccessible | kNotCoAccessible) &
                inprops2;
  }
  return outprops;
}

// Property string names (indexed by bit position).
const char* PropertyNames[] = {
    // Binary.
    "expanded", "mutable", "error", "", "", "", "", "", "", "", "", "", "", "",
    "", "",
    // Ternary.
    "acceptor", "not acceptor", "input deterministic",
    "non input deterministic", "output deterministic",
    "non output deterministic", "input/output epsilons",
    "no input/output epsilons", "input epsilons", "no input epsilons",
    "output epsilons", "no output epsilons", "input label sorted",
    "not input label sorted", "output label sorted", "not output label sorted",
    "weighted", "unweighted", "cyclic", "acyclic", "cyclic at initial state",
    "acyclic at initial state", "top sorted", "not top sorted", "accessible",
    "not accessible", "coaccessible", "not coaccessible", "string",
    "not string", "weighted cycles", "unweighted cycles"};

}  // namespace fst
