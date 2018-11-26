// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to minimize an FST.

#ifndef FST_MINIMIZE_H_
#define FST_MINIMIZE_H_

#include <cmath>

#include <algorithm>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/arcsort.h>
#include <fst/connect.h>
#include <fst/dfs-visit.h>
#include <fst/encode.h>
#include <fst/factor-weight.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/partition.h>
#include <fst/push.h>
#include <fst/queue.h>
#include <fst/reverse.h>
#include <fst/shortest-distance.h>
#include <fst/state-map.h>


namespace fst {
namespace internal {

// Comparator for creating partition.
template <class Arc>
class StateComparator {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  StateComparator(const Fst<Arc> &fst, const Partition<StateId> &partition)
      : fst_(fst), partition_(partition) {}

  // Compares state x with state y based on sort criteria.
  bool operator()(const StateId x, const StateId y) const {
    // Checks for final state equivalence.
    const auto xfinal = fst_.Final(x).Hash();
    const auto yfinal = fst_.Final(y).Hash();
    if (xfinal < yfinal) {
      return true;
    } else if (xfinal > yfinal) {
      return false;
    }
    // Checks for number of arcs.
    if (fst_.NumArcs(x) < fst_.NumArcs(y)) return true;
    if (fst_.NumArcs(x) > fst_.NumArcs(y)) return false;
    // If the number of arcs are equal, checks for arc match.
    for (ArcIterator<Fst<Arc>> aiter1(fst_, x), aiter2(fst_, y);
         !aiter1.Done() && !aiter2.Done(); aiter1.Next(), aiter2.Next()) {
      const auto &arc1 = aiter1.Value();
      const auto &arc2 = aiter2.Value();
      if (arc1.ilabel < arc2.ilabel) return true;
      if (arc1.ilabel > arc2.ilabel) return false;
      if (partition_.ClassId(arc1.nextstate) <
          partition_.ClassId(arc2.nextstate))
        return true;
      if (partition_.ClassId(arc1.nextstate) >
          partition_.ClassId(arc2.nextstate))
        return false;
    }
    return false;
  }

 private:
  const Fst<Arc> &fst_;
  const Partition<StateId> &partition_;
};

// Computes equivalence classes for cyclic unweighted acceptors. For cyclic
// minimization we use the classic Hopcroft minimization algorithm, which has
// complexity O(E log V) where E is the number of arcs and V is the number of
// states.
//
// For more information, see:
//
//  Hopcroft, J. 1971. An n Log n algorithm for minimizing states in a finite
//  automaton. Ms, Stanford University.
//
// Note: the original presentation of the paper was for a finite automaton (==
// deterministic, unweighted acceptor), but we also apply it to the
// nondeterministic case, where it is also applicable as long as the semiring is
// idempotent (if the semiring is not idempotent, there are some complexities
// in keeping track of the weight when there are multiple arcs to states that
// will be merged, and we don't deal with this).
template <class Arc, class Queue>
class CyclicMinimizer {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using ClassId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using RevArc = ReverseArc<Arc>;

  explicit CyclicMinimizer(const ExpandedFst<Arc> &fst) {
    Initialize(fst);
    Compute(fst);
  }

  const Partition<StateId> &GetPartition() const { return P_; }

 private:
  // StateILabelHasher is a hashing object that computes a hash-function
  // of an FST state that depends only on the set of ilabels on arcs leaving
  // the state [note: it assumes that the arcs are ilabel-sorted].
  // In order to work correctly for non-deterministic automata, multiple
  // instances of the same ilabel count the same as a single instance.
  class StateILabelHasher {
   public:
    explicit StateILabelHasher(const Fst<Arc> &fst) : fst_(fst) {}

    using Label = typename Arc::Label;
    using StateId = typename Arc::StateId;

    size_t operator()(const StateId s) {
      const size_t p1 = 7603;
      const size_t p2 = 433024223;
      size_t result = p2;
      size_t current_ilabel = kNoLabel;
      for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
        Label this_ilabel = aiter.Value().ilabel;
        if (this_ilabel != current_ilabel) {  // Ignores repeats.
          result = p1 * result + this_ilabel;
          current_ilabel = this_ilabel;
        }
      }
      return result;
    }

   private:
    const Fst<Arc> &fst_;
  };

  class ArcIterCompare {
   public:
    explicit ArcIterCompare(const Partition<StateId> &partition)
        : partition_(partition) {}

    ArcIterCompare(const ArcIterCompare &comp) : partition_(comp.partition_) {}

    // Compares two iterators based on their input labels.
    bool operator()(const ArcIterator<Fst<RevArc>> *x,
                    const ArcIterator<Fst<RevArc>> *y) const {
      const auto &xarc = x->Value();
      const auto &yarc = y->Value();
      return xarc.ilabel > yarc.ilabel;
    }

   private:
    const Partition<StateId> &partition_;
  };

  using ArcIterQueue =
      std::priority_queue<ArcIterator<Fst<RevArc>> *,
                          std::vector<ArcIterator<Fst<RevArc>> *>,
                          ArcIterCompare>;

 private:
  // Prepartitions the space into equivalence classes. We ensure that final and
  // non-final states always go into different equivalence classes, and we use
  // class StateILabelHasher to make sure that most of the time, states with
  // different sets of ilabels on arcs leaving them, go to different partitions.
  // Note: for the O(n) guarantees we don't rely on the goodness of this
  // hashing function---it just provides a bonus speedup.
  void PrePartition(const ExpandedFst<Arc> &fst) {
    VLOG(5) << "PrePartition";
    StateId next_class = 0;
    auto num_states = fst.NumStates();
    // Allocates a temporary vector to store the initial class mappings, so that
    // we can allocate the classes all at once.
    std::vector<StateId> state_to_initial_class(num_states);
    {
      // We maintain two maps from hash-value to class---one for final states
      // (final-prob == One()) and one for non-final states
      // (final-prob == Zero()). We are processing unweighted acceptors, so the
      // are the only two possible values.
      using HashToClassMap = std::unordered_map<size_t, StateId>;
      HashToClassMap hash_to_class_nonfinal;
      HashToClassMap hash_to_class_final;
      StateILabelHasher hasher(fst);
      for (StateId s = 0; s < num_states; ++s) {
        size_t hash = hasher(s);
        HashToClassMap &this_map =
            (fst.Final(s) != Weight::Zero() ? hash_to_class_final
                                            : hash_to_class_nonfinal);
        // Avoids two map lookups by using 'insert' instead of 'find'.
        auto p = this_map.insert(std::make_pair(hash, next_class));
        state_to_initial_class[s] = p.second ? next_class++ : p.first->second;
      }
      // Lets the unordered_maps go out of scope before we allocate the classes,
      // to reduce the maximum amount of memory used.
    }
    P_.AllocateClasses(next_class);
    for (StateId s = 0; s < num_states; ++s) {
      P_.Add(s, state_to_initial_class[s]);
    }
    for (StateId c = 0; c < next_class; ++c) L_.Enqueue(c);
    VLOG(5) << "Initial Partition: " << P_.NumClasses();
  }

  // Creates inverse transition Tr_ = rev(fst), loops over states in FST and
  // splits on final, creating two blocks in the partition corresponding to
  // final, non-final.
  void Initialize(const ExpandedFst<Arc> &fst) {
    // Constructs Tr.
    Reverse(fst, &Tr_);
    ILabelCompare<RevArc> ilabel_comp;
    ArcSort(&Tr_, ilabel_comp);
    // Tells the partition how many elements to allocate. The first state in
    // Tr_ is super-final state.
    P_.Initialize(Tr_.NumStates() - 1);
    // Prepares initial partition.
    PrePartition(fst);
    // Allocates arc iterator queue.
    ArcIterCompare comp(P_);
    aiter_queue_.reset(new ArcIterQueue(comp));
  }
  // Partitions all classes with destination C.
  void Split(ClassId C) {
    // Prepares priority queue: opens arc iterator for each state in C, and
    // inserts into priority queue.
    for (PartitionIterator<StateId> siter(P_, C); !siter.Done(); siter.Next()) {
      StateId s = siter.Value();
      if (Tr_.NumArcs(s + 1)) {
        aiter_queue_->push(new ArcIterator<Fst<RevArc>>(Tr_, s + 1));
      }
    }
    // Now pops arc iterator from queue, splits entering equivalence class, and
    // re-inserts updated iterator into queue.
    Label prev_label = -1;
    while (!aiter_queue_->empty()) {
      std::unique_ptr<ArcIterator<Fst<RevArc>>> aiter(aiter_queue_->top());
      aiter_queue_->pop();
      if (aiter->Done()) continue;
      const auto &arc = aiter->Value();
      auto from_state = aiter->Value().nextstate - 1;
      auto from_label = arc.ilabel;
      if (prev_label != from_label) P_.FinalizeSplit(&L_);
      auto from_class = P_.ClassId(from_state);
      if (P_.ClassSize(from_class) > 1) P_.SplitOn(from_state);
      prev_label = from_label;
      aiter->Next();
      if (!aiter->Done()) aiter_queue_->push(aiter.release());
    }
    P_.FinalizeSplit(&L_);
  }

  // Main loop for Hopcroft minimization.
  void Compute(const Fst<Arc> &fst) {
    // Processes active classes (FIFO, or FILO).
    while (!L_.Empty()) {
      const auto C = L_.Head();
      L_.Dequeue();
      Split(C);  // Splits on C, all labels in C.
    }
  }

 private:
  // Partioning of states into equivalence classes.
  Partition<StateId> P_;
  // Set of active classes to be processed in partition P.
  Queue L_;
  // Reverses transition function.
  VectorFst<RevArc> Tr_;
  // Priority queue of open arc iterators for all states in the splitter
  // equivalence class.
  std::unique_ptr<ArcIterQueue> aiter_queue_;
};

// Computes equivalence classes for acyclic FST.
//
// Complexity:
//
//   O(E)
//
// where E is the number of arcs.
//
// For more information, see:
//
// Revuz, D. 1992. Minimization of acyclic deterministic automata in linear
// time. Theoretical Computer Science 92(1): 181-189.
template <class Arc>
class AcyclicMinimizer {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using ClassId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit AcyclicMinimizer(const ExpandedFst<Arc> &fst) {
    Initialize(fst);
    Refine(fst);
  }

  const Partition<StateId> &GetPartition() { return partition_; }

 private:
  // DFS visitor to compute the height (distance) to final state.
  class HeightVisitor {
   public:
    HeightVisitor() : max_height_(0), num_states_(0) {}

    // Invoked before DFS visit.
    void InitVisit(const Fst<Arc> &fst) {}

    // Invoked when state is discovered (2nd arg is DFS tree root).
    bool InitState(StateId s, StateId root) {
      // Extends height array and initialize height (distance) to 0.
      for (StateId i = height_.size(); i <= s; ++i) height_.push_back(-1);
      if (s >= num_states_) num_states_ = s + 1;
      return true;
    }

    // Invoked when tree arc examined (to undiscovered state).
    bool TreeArc(StateId s, const Arc &arc) { return true; }

    // Invoked when back arc examined (to unfinished state).
    bool BackArc(StateId s, const Arc &arc) { return true; }

    // Invoked when forward or cross arc examined (to finished state).
    bool ForwardOrCrossArc(StateId s, const Arc &arc) {
      if (height_[arc.nextstate] + 1 > height_[s]) {
        height_[s] = height_[arc.nextstate] + 1;
      }
      return true;
    }

    // Invoked when state finished (parent is kNoStateId for tree root).
    void FinishState(StateId s, StateId parent, const Arc *parent_arc) {
      if (height_[s] == -1) height_[s] = 0;
      const auto h = height_[s] + 1;
      if (parent >= 0) {
        if (h > height_[parent]) height_[parent] = h;
        if (h > max_height_) max_height_ = h;
      }
    }

    // Invoked after DFS visit.
    void FinishVisit() {}

    size_t max_height() const { return max_height_; }

    const std::vector<StateId> &height() const { return height_; }

    size_t num_states() const { return num_states_; }

   private:
    std::vector<StateId> height_;
    size_t max_height_;
    size_t num_states_;
  };

 private:
  // Cluster states according to height (distance to final state)
  void Initialize(const Fst<Arc> &fst) {
    // Computes height (distance to final state).
    HeightVisitor hvisitor;
    DfsVisit(fst, &hvisitor);
    // Creates initial partition based on height.
    partition_.Initialize(hvisitor.num_states());
    partition_.AllocateClasses(hvisitor.max_height() + 1);
    const auto &hstates = hvisitor.height();
    for (StateId s = 0; s < hstates.size(); ++s) partition_.Add(s, hstates[s]);
  }

  // Refines states based on arc sort (out degree, arc equivalence).
  void Refine(const Fst<Arc> &fst) {
    using EquivalenceMap = std::map<StateId, StateId, StateComparator<Arc>>;
    StateComparator<Arc> comp(fst, partition_);
    // Starts with tail (height = 0).
    auto height = partition_.NumClasses();
    for (StateId h = 0; h < height; ++h) {
      EquivalenceMap equiv_classes(comp);
      // Sorts states within equivalence class.
      PartitionIterator<StateId> siter(partition_, h);
      equiv_classes[siter.Value()] = h;
      for (siter.Next(); !siter.Done(); siter.Next()) {
        auto insert_result =
            equiv_classes.insert(std::make_pair(siter.Value(), kNoStateId));
        if (insert_result.second) {
          insert_result.first->second = partition_.AddClass();
        }
      }
      // Creates refined partition.
      for (siter.Reset(); !siter.Done();) {
        const auto s = siter.Value();
        const auto old_class = partition_.ClassId(s);
        const auto new_class = equiv_classes[s];
        // A move operation can invalidate the iterator, so we first update
        // the iterator to the next element before we move the current element
        // out of the list.
        siter.Next();
        if (old_class != new_class) partition_.Move(s, new_class);
      }
    }
  }

 private:
  Partition<StateId> partition_;
};

// Given a partition and a Mutable FST, merges states of Fst in place (i.e.,
// destructively). Merging works by taking the first state in a class of the
// partition to be the representative state for the class. Each arc is then
// reconnected to this state. All states in the class are merged by adding
// their arcs to the representative state.
template <class Arc>
void MergeStates(const Partition<typename Arc::StateId> &partition,
                 MutableFst<Arc> *fst) {
  using StateId = typename Arc::StateId;
  std::vector<StateId> state_map(partition.NumClasses());
  for (StateId i = 0; i < partition.NumClasses(); ++i) {
    PartitionIterator<StateId> siter(partition, i);
    state_map[i] = siter.Value();  // First state in partition.
  }
  // Relabels destination states.
  for (StateId c = 0; c < partition.NumClasses(); ++c) {
    for (PartitionIterator<StateId> siter(partition, c); !siter.Done();
         siter.Next()) {
      const auto s = siter.Value();
      for (MutableArcIterator<MutableFst<Arc>> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        auto arc = aiter.Value();
        arc.nextstate = state_map[partition.ClassId(arc.nextstate)];
        if (s == state_map[c]) {  // For the first state, just sets destination.
          aiter.SetValue(arc);
        } else {
          fst->AddArc(state_map[c], arc);
        }
      }
    }
  }
  fst->SetStart(state_map[partition.ClassId(fst->Start())]);
  Connect(fst);
}

template <class Arc>
void AcceptorMinimize(MutableFst<Arc> *fst,
                      bool allow_acyclic_minimization = true) {
  if (!(fst->Properties(kAcceptor | kUnweighted, true) ==
        (kAcceptor | kUnweighted))) {
    FSTERROR() << "FST is not an unweighted acceptor";
    fst->SetProperties(kError, kError);
    return;
  }
  // Connects FST before minimization, handles disconnected states.
  Connect(fst);
  if (fst->NumStates() == 0) return;
  if (allow_acyclic_minimization && fst->Properties(kAcyclic, true)) {
    // Acyclic minimization (Revuz).
    VLOG(2) << "Acyclic minimization";
    ArcSort(fst, ILabelCompare<Arc>());
    AcyclicMinimizer<Arc> minimizer(*fst);
    MergeStates(minimizer.GetPartition(), fst);
  } else {
    // Either the FST has cycles, or it's generated from non-deterministic input
    // (which the Revuz algorithm can't handle), so use the cyclic minimization
    // algorithm of Hopcroft.
    VLOG(2) << "Cyclic minimization";
    CyclicMinimizer<Arc, LifoQueue<typename Arc::StateId>> minimizer(*fst);
    MergeStates(minimizer.GetPartition(), fst);
  }
  // Merges in appropriate semiring
  ArcUniqueMapper<Arc> mapper(*fst);
  StateMap(fst, mapper);
}

}  // namespace internal

// In place minimization of deterministic weighted automata and transducers,
// and also non-deterministic ones if they use an idempotent semiring.
// For transducers, if the 'sfst' argument is not null, the algorithm
// produces a compact factorization of the minimal transducer.
//
// In the acyclic deterministic case, we use an algorithm from Revuz that is
// linear in the number of arcs (edges) in the machine.
//
// In the cyclic or non-deterministic case, we use the classical Hopcroft
// minimization (which was presented for the deterministic case but which
// also works for non-deterministic FSTs); this has complexity O(e log v).
//
template <class Arc>
void Minimize(MutableFst<Arc> *fst, MutableFst<Arc> *sfst = nullptr,
              float delta = kShortestDelta, bool allow_nondet = false) {
  using Weight = typename Arc::Weight;
  const auto props = fst->Properties(
      kAcceptor | kIDeterministic | kWeighted | kUnweighted, true);
  bool allow_acyclic_minimization;
  if (props & kIDeterministic) {
    allow_acyclic_minimization = true;
  } else {
    // Our approach to minimization of non-deterministic FSTs will only work in
    // idempotent semirings---for non-deterministic inputs, a state could have
    // multiple transitions to states that will get merged, and we'd have to
    // sum their weights. The algorithm doesn't handle that.
    if (!(Weight::Properties() & kIdempotent)) {
      fst->SetProperties(kError, kError);
      FSTERROR() << "Cannot minimize a non-deterministic FST over a "
                    "non-idempotent semiring";
      return;
    } else if (!allow_nondet) {
      fst->SetProperties(kError, kError);
      FSTERROR() << "Refusing to minimize a non-deterministic FST with "
                 << "allow_nondet = false";
      return;
    }
    // The Revuz algorithm won't work for nondeterministic inputs, so if the
    // input is nondeterministic, we'll have to pass a bool saying not to use
    // that algorithm. We check at this level rather than in AcceptorMinimize(),
    // because it's possible that the FST at this level could be deterministic,
    // but a harmless type of non-determinism could be introduced by Encode()
    // (thanks to kEncodeWeights, if the FST has epsilons and has a final
    // weight with weights equal to some epsilon arc.)
    allow_acyclic_minimization = false;
  }
  if (!(props & kAcceptor)) {  // Weighted transducer.
    VectorFst<GallicArc<Arc, GALLIC_LEFT>> gfst;
    ArcMap(*fst, &gfst, ToGallicMapper<Arc, GALLIC_LEFT>());
    fst->DeleteStates();
    gfst.SetProperties(kAcceptor, kAcceptor);
    Push(&gfst, REWEIGHT_TO_INITIAL, delta);
    ArcMap(&gfst, QuantizeMapper<GallicArc<Arc, GALLIC_LEFT>>(delta));
    EncodeMapper<GallicArc<Arc, GALLIC_LEFT>> encoder(
        kEncodeLabels | kEncodeWeights, ENCODE);
    Encode(&gfst, &encoder);
    internal::AcceptorMinimize(&gfst, allow_acyclic_minimization);
    Decode(&gfst, encoder);
    if (!sfst) {
      FactorWeightFst<GallicArc<Arc, GALLIC_LEFT>,
                      GallicFactor<typename Arc::Label, Weight, GALLIC_LEFT>>
          fwfst(gfst);
      std::unique_ptr<SymbolTable> osyms(
          fst->OutputSymbols() ? fst->OutputSymbols()->Copy() : nullptr);
      ArcMap(fwfst, fst, FromGallicMapper<Arc, GALLIC_LEFT>());
      fst->SetOutputSymbols(osyms.get());
    } else {
      sfst->SetOutputSymbols(fst->OutputSymbols());
      GallicToNewSymbolsMapper<Arc, GALLIC_LEFT> mapper(sfst);
      ArcMap(gfst, fst, &mapper);
      fst->SetOutputSymbols(sfst->InputSymbols());
    }
  } else if (props & kWeighted) {  // Weighted acceptor.
    Push(fst, REWEIGHT_TO_INITIAL, delta);
    ArcMap(fst, QuantizeMapper<Arc>(delta));
    EncodeMapper<Arc> encoder(kEncodeLabels | kEncodeWeights, ENCODE);
    Encode(fst, &encoder);
    internal::AcceptorMinimize(fst, allow_acyclic_minimization);
    Decode(fst, encoder);
  } else {  // Unweighted acceptor.
    internal::AcceptorMinimize(fst, allow_acyclic_minimization);
  }
}

}  // namespace fst

#endif  // FST_MINIMIZE_H_
