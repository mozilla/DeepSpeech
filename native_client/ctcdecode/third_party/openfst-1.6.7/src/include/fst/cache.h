// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// An FST implementation that caches FST elements of a delayed computation.

#ifndef FST_CACHE_H_
#define FST_CACHE_H_

#include <functional>
#include <unordered_map>
using std::unordered_map;
using std::unordered_multimap;
#include <list>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/vector-fst.h>


DECLARE_bool(fst_default_cache_gc);
DECLARE_int64(fst_default_cache_gc_limit);

namespace fst {

// Options for controlling caching behavior; higher level than CacheImplOptions.
struct CacheOptions {
  bool gc;          // Enables GC.
  size_t gc_limit;  // Number of bytes allowed before GC.

  explicit CacheOptions(bool gc = FLAGS_fst_default_cache_gc,
                        size_t gc_limit = FLAGS_fst_default_cache_gc_limit)
      : gc(gc), gc_limit(gc_limit) {}
};

// Options for controlling caching behavior, at a lower level than
// CacheOptions; templated on the cache store and allows passing the store.
template <class CacheStore>
struct CacheImplOptions {
  bool gc;            // Enables GC.
  size_t gc_limit;    // Number of bytes allowed before GC.
  CacheStore *store;  // Cache store.
  bool own_store;     // Should CacheImpl takes ownership of the store?

  explicit CacheImplOptions(bool gc = FLAGS_fst_default_cache_gc,
                            size_t gc_limit = FLAGS_fst_default_cache_gc_limit,
                            CacheStore *store = nullptr)
      : gc(gc), gc_limit(gc_limit), store(store), own_store(true) {}

  explicit CacheImplOptions(const CacheOptions &opts)
      : gc(opts.gc), gc_limit(opts.gc_limit), store(nullptr), own_store(true) {}
};

// Cache flags.
constexpr uint32 kCacheFinal = 0x0001;   // Final weight has been cached.
constexpr uint32 kCacheArcs = 0x0002;    // Arcs have been cached.
constexpr uint32 kCacheInit = 0x0004;    // Initialized by GC.
constexpr uint32 kCacheRecent = 0x0008;  // Visited since GC.
constexpr uint32 kCacheFlags =
    kCacheFinal | kCacheArcs | kCacheInit | kCacheRecent;

// Cache state, with arcs stored in a per-state std::vector.
template <class A, class M = PoolAllocator<A>>
class CacheState {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using ArcAllocator = M;
  using StateAllocator =
      typename ArcAllocator::template rebind<CacheState<A, M>>::other;

  // Provides STL allocator for arcs.
  explicit CacheState(const ArcAllocator &alloc)
      : final_(Weight::Zero()),
        niepsilons_(0),
        noepsilons_(0),
        arcs_(alloc),
        flags_(0),
        ref_count_(0) {}

  CacheState(const CacheState<A> &state, const ArcAllocator &alloc)
      : final_(state.Final()),
        niepsilons_(state.NumInputEpsilons()),
        noepsilons_(state.NumOutputEpsilons()),
        arcs_(state.arcs_.begin(), state.arcs_.end(), alloc),
        flags_(state.Flags()),
        ref_count_(0) {}

  void Reset() {
    final_ = Weight::Zero();
    niepsilons_ = 0;
    noepsilons_ = 0;
    ref_count_ = 0;
    flags_ = 0;
    arcs_.clear();
  }

  Weight Final() const { return final_; }

  size_t NumInputEpsilons() const { return niepsilons_; }

  size_t NumOutputEpsilons() const { return noepsilons_; }

  size_t NumArcs() const { return arcs_.size(); }

  const Arc &GetArc(size_t n) const { return arcs_[n]; }

  // Used by the ArcIterator<Fst<Arc>> efficient implementation.
  const Arc *Arcs() const { return !arcs_.empty() ? &arcs_[0] : nullptr; }

  // Accesses flags; used by the caller.
  uint32 Flags() const { return flags_; }

  // Accesses ref count; used by the caller.
  int RefCount() const { return ref_count_; }

  void SetFinal(Weight weight) { final_ = std::move(weight); }

  void ReserveArcs(size_t n) { arcs_.reserve(n); }

  // Adds one arc at a time with all needed book-keeping; use PushArc and
  // SetArcs for a more efficient alternative.
  void AddArc(const Arc &arc) {
    arcs_.push_back(arc);
    if (arc.ilabel == 0) ++niepsilons_;
    if (arc.olabel == 0) ++noepsilons_;
  }

  // Adds one arc at a time with delayed book-keeping; finalize with SetArcs().
  void PushArc(const Arc &arc) { arcs_.push_back(arc); }

  // Finalizes arcs book-keeping; call only once.
  void SetArcs() {
    for (const auto &arc : arcs_) {
      if (arc.ilabel == 0) ++niepsilons_;
      if (arc.olabel == 0) ++noepsilons_;
    }
  }

  // Modifies nth arc.
  void SetArc(const Arc &arc, size_t n) {
    if (arcs_[n].ilabel == 0) --niepsilons_;
    if (arcs_[n].olabel == 0) --noepsilons_;
    if (arc.ilabel == 0) ++niepsilons_;
    if (arc.olabel == 0) ++noepsilons_;
    arcs_[n] = arc;
  }

  // Deletes all arcs.
  void DeleteArcs() {
    niepsilons_ = 0;
    noepsilons_ = 0;
    arcs_.clear();
  }

  void DeleteArcs(size_t n) {
    for (size_t i = 0; i < n; ++i) {
      if (arcs_.back().ilabel == 0) --niepsilons_;
      if (arcs_.back().olabel == 0) --noepsilons_;
      arcs_.pop_back();
    }
  }

  // Sets status flags; used by the caller.
  void SetFlags(uint32 flags, uint32 mask) const {
    flags_ &= ~mask;
    flags_ |= flags;
  }

  // Mutates reference counts; used by the caller.

  int IncrRefCount() const { return ++ref_count_; }

  int DecrRefCount() const { return --ref_count_; }

  // Used by the ArcIterator<Fst<Arc>> efficient implementation.
  int *MutableRefCount() const { return &ref_count_; }

  // Used for state class allocation.
  void *operator new(size_t size, StateAllocator *alloc) {
    return alloc->allocate(1);
  }

  // For state destruction and memory freeing.
  static void Destroy(CacheState<Arc> *state, StateAllocator *alloc) {
    if (state) {
      state->~CacheState<Arc>();
      alloc->deallocate(state, 1);
    }
  }

 private:
  Weight final_;                         // Final weight.
  size_t niepsilons_;                    // # of input epsilons.
  size_t noepsilons_;                    // # of output epsilons.
  std::vector<Arc, ArcAllocator> arcs_;  // Arcs representation.
  mutable uint32 flags_;
  mutable int ref_count_;  // If 0, available for GC.
};

// Cache store, allocating and storing states, providing a mapping from state
// IDs to cached states, and an iterator over these states. The state template
// argument must implement the CacheState interface. The state for a StateId s
// is constructed when requested by GetMutableState(s) if it is not yet stored.
// Initially, a state has a reference count of zero, but the user may increment
// or decrement this to control the time of destruction. In particular, a state
// is destroyed when:
//
// 1. This instance is destroyed, or
// 2. Clear() or Delete() is called, or
// 3. Possibly (implementation-dependently) when:
//    - Garbage collection is enabled (as defined by opts.gc),
//    - The cache store size exceeds the limits (as defined by opts.gc_limits),
//    - The state's reference count is zero, and
//    - The state is not the most recently requested state.
//
// template <class S>
// class CacheStore {
//  public:
//   using State = S;
//   using Arc = typename State::Arc;
//   using StateId = typename Arc::StateId;
//
//   // Required constructors/assignment operators.
//   explicit CacheStore(const CacheOptions &opts);
//
//   // Returns nullptr if state is not stored.
//   const State *GetState(StateId s);
//
//   // Creates state if state is not stored.
//   State *GetMutableState(StateId s);
//
//   // Similar to State::AddArc() but updates cache store book-keeping.
//   void AddArc(State *state, const Arc &arc);
//
//   // Similar to State::SetArcs() but updates cache store book-keeping; call
//   // only once.
//   void SetArcs(State *state);
//
//   // Similar to State::DeleteArcs() but updates cache store book-keeping.
//
//   void DeleteArcs(State *state);
//
//   void DeleteArcs(State *state, size_t n);
//
//   // Deletes all cached states.
//   void Clear();
//
//   // Iterates over cached states (in an arbitrary order); only needed if
//   // opts.gc is true.
//   bool Done() const;      // End of iteration.
//   StateId Value() const;  // Current state.
//   void Next();            // Advances to next state (when !Done).
//   void Reset();           // Returns to initial condition.
//   void Delete();          // Deletes current state and advances to next.
// };

// Container cache stores.

// This class uses a vector of pointers to states to store cached states.
template <class S>
class VectorCacheStore {
 public:
  using State = S;
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;
  using StateList = std::list<StateId, PoolAllocator<StateId>>;

  // Required constructors/assignment operators.
  explicit VectorCacheStore(const CacheOptions &opts) : cache_gc_(opts.gc) {
    Clear();
    Reset();
  }

  VectorCacheStore(const VectorCacheStore<S> &store)
      : cache_gc_(store.cache_gc_) {
    CopyStates(store);
    Reset();
  }

  ~VectorCacheStore() { Clear(); }

  VectorCacheStore<State> &operator=(const VectorCacheStore<State> &store) {
    if (this != &store) {
      CopyStates(store);
      Reset();
    }
    return *this;
  }

  // Returns nullptr if state is not stored.
  const State *GetState(StateId s) const {
    return s < state_vec_.size() ? state_vec_[s] : nullptr;
  }

  // Creates state if state is not stored.
  State *GetMutableState(StateId s) {
    State *state = nullptr;
    if (s >= state_vec_.size()) {
      state_vec_.resize(s + 1, nullptr);
    } else {
      state = state_vec_[s];
    }
    if (!state) {
      state = new (&state_alloc_) State(arc_alloc_);
      state_vec_[s] = state;
      if (cache_gc_) state_list_.push_back(s);
    }
    return state;
  }

  // Similar to State::AddArc() but updates cache store book-keeping
  void AddArc(State *state, const Arc &arc) { state->AddArc(arc); }

  // Similar to State::SetArcs() but updates cache store book-keeping; call
  // only once.
  void SetArcs(State *state) { state->SetArcs(); }

  // Deletes all arcs.
  void DeleteArcs(State *state) { state->DeleteArcs(); }

  // Deletes some arcs.
  void DeleteArcs(State *state, size_t n) { state->DeleteArcs(n); }

  // Deletes all cached states.
  void Clear() {
    for (StateId s = 0; s < state_vec_.size(); ++s) {
      State::Destroy(state_vec_[s], &state_alloc_);
    }
    state_vec_.clear();
    state_list_.clear();
  }

  // Iterates over cached states (in an arbitrary order); only works if GC is
  // enabled (o.w. avoiding state_list_ overhead).
  bool Done() const { return iter_ == state_list_.end(); }

  StateId Value() const { return *iter_; }

  void Next() { ++iter_; }

  void Reset() { iter_ = state_list_.begin(); }

  // Deletes current state and advances to next.
  void Delete() {
    State::Destroy(state_vec_[*iter_], &state_alloc_);
    state_vec_[*iter_] = nullptr;
    state_list_.erase(iter_++);
  }

 private:
  void CopyStates(const VectorCacheStore<State> &store) {
    Clear();
    state_vec_.reserve(store.state_vec_.size());
    for (StateId s = 0; s < store.state_vec_.size(); ++s) {
      State *state = nullptr;
      const auto *store_state = store.state_vec_[s];
      if (store_state) {
        state = new (&state_alloc_) State(*store_state, arc_alloc_);
        if (cache_gc_) state_list_.push_back(s);
      }
      state_vec_.push_back(state);
    }
  }

  bool cache_gc_;                               // Supports iteration when true.
  std::vector<State *> state_vec_;              // Vector of states (or null).
  StateList state_list_;                        // List of states.
  typename StateList::iterator iter_;           // State list iterator.
  typename State::StateAllocator state_alloc_;  // For state allocation.
  typename State::ArcAllocator arc_alloc_;      // For arc allocation.
};

// This class uses a hash map from state IDs to pointers to cached states.
template <class S>
class HashCacheStore {
 public:
  using State = S;
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;

  using StateMap =
      std::unordered_map<StateId, State *, std::hash<StateId>,
                         std::equal_to<StateId>,
                         PoolAllocator<std::pair<const StateId, State *>>>;

  // Required constructors/assignment operators.
  explicit HashCacheStore(const CacheOptions &opts) {
    Clear();
    Reset();
  }

  HashCacheStore(const HashCacheStore<S> &store) {
    CopyStates(store);
    Reset();
  }

  ~HashCacheStore() { Clear(); }

  HashCacheStore<State> &operator=(const HashCacheStore<State> &store) {
    if (this != &store) {
      CopyStates(store);
      Reset();
    }
    return *this;
  }

  // Returns nullptr if state is not stored.
  const State *GetState(StateId s) const {
    const auto it = state_map_.find(s);
    return it != state_map_.end() ? it->second : nullptr;
  }

  // Creates state if state is not stored.
  State *GetMutableState(StateId s) {
    auto *&state = state_map_[s];
    if (!state) state = new (&state_alloc_) State(arc_alloc_);
    return state;
  }

  // Similar to State::AddArc() but updates cache store book-keeping.
  void AddArc(State *state, const Arc &arc) { state->AddArc(arc); }

  // Similar to State::SetArcs() but updates internal cache size; call only
  // once.
  void SetArcs(State *state) { state->SetArcs(); }

  // Deletes all arcs.
  void DeleteArcs(State *state) { state->DeleteArcs(); }

  // Deletes some arcs.
  void DeleteArcs(State *state, size_t n) { state->DeleteArcs(n); }

  // Deletes all cached states.
  void Clear() {
    for (auto it = state_map_.begin(); it != state_map_.end(); ++it) {
      State::Destroy(it->second, &state_alloc_);
    }
    state_map_.clear();
  }

  // Iterates over cached states (in an arbitrary order).
  bool Done() const { return iter_ == state_map_.end(); }

  StateId Value() const { return iter_->first; }

  void Next() { ++iter_; }

  void Reset() { iter_ = state_map_.begin(); }

  // Deletes current state and advances to next.
  void Delete() {
    State::Destroy(iter_->second, &state_alloc_);
    state_map_.erase(iter_++);
  }

 private:
  void CopyStates(const HashCacheStore<State> &store) {
    Clear();
    for (auto it = store.state_map_.begin(); it != store.state_map_.end();
         ++it) {
      state_map_[it->first] =
          new (&state_alloc_) State(*it->second, arc_alloc_);
    }
  }

  StateMap state_map_;                          // Map from state ID to state.
  typename StateMap::iterator iter_;            // State map iterator.
  typename State::StateAllocator state_alloc_;  // For state allocation.
  typename State::ArcAllocator arc_alloc_;      // For arc allocation.
};

// Garbage-colllection cache stores.

// This class implements a simple garbage collection scheme when
// 'opts.gc_limit = 0'. In particular, the first cached state is reused for each
// new state so long as the reference count is zero on the to-be-reused state.
// Otherwise, the full underlying store is used. The caller can increment the
// reference count to inhibit the GC of in-use states (e.g., in an ArcIterator).
//
// The typical use case for this optimization is when a single pass over a
// cached
// FST is performed with only one-state expanded at a time.
template <class CacheStore>
class FirstCacheStore {
 public:
  using State = typename CacheStore::State;
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;

  // Required constructors/assignment operators.
  explicit FirstCacheStore(const CacheOptions &opts)
      : store_(opts),
        cache_gc_(opts.gc_limit == 0),  // opts.gc ignored historically.
        cache_first_state_id_(kNoStateId),
        cache_first_state_(nullptr) {}

  FirstCacheStore(const FirstCacheStore<CacheStore> &store)
      : store_(store.store_),
        cache_gc_(store.cache_gc_),
        cache_first_state_id_(store.cache_first_state_id_),
        cache_first_state_(store.cache_first_state_id_ != kNoStateId
                               ? store_.GetMutableState(0)
                               : nullptr) {}

  FirstCacheStore<CacheStore> &operator=(
      const FirstCacheStore<CacheStore> &store) {
    if (this != &store) {
      store_ = store.store_;
      cache_gc_ = store.cache_gc_;
      cache_first_state_id_ = store.cache_first_state_id_;
      cache_first_state_ = store.cache_first_state_id_ != kNoStateId
                               ? store_.GetMutableState(0)
                               : nullptr;
    }
    return *this;
  }

  // Returns nullptr if state is not stored.
  const State *GetState(StateId s) const {
    // store_ state 0 may hold first cached state; the rest are shifted by 1.
    return s == cache_first_state_id_ ? cache_first_state_
                                      : store_.GetState(s + 1);
  }

  // Creates state if state is not stored.
  State *GetMutableState(StateId s) {
    // store_ state 0 used to hold first cached state; the rest are shifted by
    // 1.
    if (cache_first_state_id_ == s) {
      return cache_first_state_;  // Request for first cached state.
    }
    if (cache_gc_) {
      if (cache_first_state_id_ == kNoStateId) {
        cache_first_state_id_ = s;  // Sets first cached state.
        cache_first_state_ = store_.GetMutableState(0);
        cache_first_state_->SetFlags(kCacheInit, kCacheInit);
        cache_first_state_->ReserveArcs(2 * kAllocSize);
        return cache_first_state_;
      } else if (cache_first_state_->RefCount() == 0) {
        cache_first_state_id_ = s;  // Updates first cached state.
        cache_first_state_->Reset();
        cache_first_state_->SetFlags(kCacheInit, kCacheInit);
        return cache_first_state_;
      } else {  // Keeps first cached state.
        cache_first_state_->SetFlags(0, kCacheInit);  // Clears initialized bit.
        cache_gc_ = false;                            // Disables GC.
      }
    }
    auto *state = store_.GetMutableState(s + 1);
    return state;
  }

  // Similar to State::AddArc() but updates cache store book-keeping.
  void AddArc(State *state, const Arc &arc) { store_.AddArc(state, arc); }

  // Similar to State::SetArcs() but updates internal cache size; call only
  // once.
  void SetArcs(State *state) { store_.SetArcs(state); }

  // Deletes all arcs
  void DeleteArcs(State *state) { store_.DeleteArcs(state); }

  // Deletes some arcs
  void DeleteArcs(State *state, size_t n) { store_.DeleteArcs(state, n); }

  // Deletes all cached states
  void Clear() {
    store_.Clear();
    cache_first_state_id_ = kNoStateId;
    cache_first_state_ = nullptr;
  }

  // Iterates over cached states (in an arbitrary order). Only needed if GC is
  // enabled.
  bool Done() const { return store_.Done(); }

  StateId Value() const {
    // store_ state 0 may hold first cached state; rest shifted + 1.
    const auto s = store_.Value();
    return s ? s - 1 : cache_first_state_id_;
  }

  void Next() { store_.Next(); }

  void Reset() { store_.Reset(); }

  // Deletes current state and advances to next.
  void Delete() {
    if (Value() == cache_first_state_id_) {
      cache_first_state_id_ = kNoStateId;
      cache_first_state_ = nullptr;
    }
    store_.Delete();
  }

 private:
  CacheStore store_;              // Underlying store.
  bool cache_gc_;                 // GC enabled.
  StateId cache_first_state_id_;  // First cached state ID.
  State *cache_first_state_;      // First cached state.
};

// This class implements mark-sweep garbage collection on an underlying cache
// store. If GC is enabled, garbage collection of states is performed in a
// rough approximation of LRU order once when 'gc_limit' bytes is reached. The
// caller can increment the reference count to inhibit the GC of in-use state
// (e.g., in an ArcIterator). With GC enabled, the 'gc_limit' parameter allows
// the caller to trade-off time vs. space.
template <class CacheStore>
class GCCacheStore {
 public:
  using State = typename CacheStore::State;
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;

  // Required constructors/assignment operators.
  explicit GCCacheStore(const CacheOptions &opts)
      : store_(opts),
        cache_gc_request_(opts.gc),
        cache_limit_(opts.gc_limit > kMinCacheLimit ? opts.gc_limit
                                                    : kMinCacheLimit),
        cache_gc_(false),
        cache_size_(0) {}

  // Returns 0 if state is not stored.
  const State *GetState(StateId s) const { return store_.GetState(s); }

  // Creates state if state is not stored
  State *GetMutableState(StateId s) {
    auto *state = store_.GetMutableState(s);
    if (cache_gc_request_ && !(state->Flags() & kCacheInit)) {
      state->SetFlags(kCacheInit, kCacheInit);
      cache_size_ += sizeof(State) + state->NumArcs() * sizeof(Arc);
      // GC is enabled once an uninited state (from underlying store) is seen.
      cache_gc_ = true;
      if (cache_size_ > cache_limit_) GC(state, false);
    }
    return state;
  }

  // Similar to State::AddArc() but updates cache store book-keeping.
  void AddArc(State *state, const Arc &arc) {
    store_.AddArc(state, arc);
    if (cache_gc_ && (state->Flags() & kCacheInit)) {
      cache_size_ += sizeof(Arc);
      if (cache_size_ > cache_limit_) GC(state, false);
    }
  }

  // Similar to State::SetArcs() but updates internal cache size; call only
  // once.
  void SetArcs(State *state) {
    store_.SetArcs(state);
    if (cache_gc_ && (state->Flags() & kCacheInit)) {
      cache_size_ += state->NumArcs() * sizeof(Arc);
      if (cache_size_ > cache_limit_) GC(state, false);
    }
  }

  // Deletes all arcs.
  void DeleteArcs(State *state) {
    if (cache_gc_ && (state->Flags() & kCacheInit)) {
      cache_size_ -= state->NumArcs() * sizeof(Arc);
    }
    store_.DeleteArcs(state);
  }

  // Deletes some arcs.
  void DeleteArcs(State *state, size_t n) {
    if (cache_gc_ && (state->Flags() & kCacheInit)) {
      cache_size_ -= n * sizeof(Arc);
    }
    store_.DeleteArcs(state, n);
  }

  // Deletes all cached states.
  void Clear() {
    store_.Clear();
    cache_size_ = 0;
  }

  // Iterates over cached states (in an arbitrary order); only needed if GC is
  // enabled.
  bool Done() const { return store_.Done(); }

  StateId Value() const { return store_.Value(); }

  void Next() { store_.Next(); }

  void Reset() { store_.Reset(); }

  // Deletes current state and advances to next.
  void Delete() {
    if (cache_gc_) {
      const auto *state = store_.GetState(Value());
      if (state->Flags() & kCacheInit) {
        cache_size_ -= sizeof(State) + state->NumArcs() * sizeof(Arc);
      }
    }
    store_.Delete();
  }

  // Removes from the cache store (not referenced-counted and not the current)
  // states that have not been accessed since the last GC until at most
  // cache_fraction * cache_limit_ bytes are cached. If that fails to free
  // enough, attempts to uncaching recently visited states as well. If still
  // unable to free enough memory, then widens cache_limit_.
  void GC(const State *current, bool free_recent, float cache_fraction = 0.666);

  // Returns the current cache size in bytes or 0 if GC is disabled.
  size_t CacheSize() const { return cache_size_; }

  // Returns the cache limit in bytes.
  size_t CacheLimit() const { return cache_limit_; }

 private:
  static constexpr size_t kMinCacheLimit = 8096;  // Minimum cache limit.

  CacheStore store_;       // Underlying store.
  bool cache_gc_request_;  // GC requested but possibly not yet enabled.
  size_t cache_limit_;     // Number of bytes allowed before GC.
  bool cache_gc_;          // GC enabled
  size_t cache_size_;      // Number of bytes cached.
};

template <class CacheStore>
void GCCacheStore<CacheStore>::GC(const State *current, bool free_recent,
                                  float cache_fraction) {
  if (!cache_gc_) return;
  VLOG(2) << "GCCacheStore: Enter GC: object = "
          << "(" << this << "), free recently cached = " << free_recent
          << ", cache size = " << cache_size_
          << ", cache frac = " << cache_fraction
          << ", cache limit = " << cache_limit_ << "\n";
  size_t cache_target = cache_fraction * cache_limit_;
  store_.Reset();
  while (!store_.Done()) {
    auto *state = store_.GetMutableState(store_.Value());
    if (cache_size_ > cache_target && state->RefCount() == 0 &&
        (free_recent || !(state->Flags() & kCacheRecent)) && state != current) {
      if (state->Flags() & kCacheInit) {
        size_t size = sizeof(State) + state->NumArcs() * sizeof(Arc);
        if (size < cache_size_) {
          cache_size_ -= size;
        }
      }
      store_.Delete();
    } else {
      state->SetFlags(0, kCacheRecent);
      store_.Next();
    }
  }
  if (!free_recent && cache_size_ > cache_target) {  // Recurses on recent.
    GC(current, true, cache_fraction);
  } else if (cache_target > 0) {  // Widens cache limit.
    while (cache_size_ > cache_target) {
      cache_limit_ *= 2;
      cache_target *= 2;
    }
  } else if (cache_size_ > 0) {
    FSTERROR() << "GCCacheStore:GC: Unable to free all cached states";
  }
  VLOG(2) << "GCCacheStore: Exit GC: object = "
          << "(" << this << "), free recently cached = " << free_recent
          << ", cache size = " << cache_size_
          << ", cache frac = " << cache_fraction
          << ", cache limit = " << cache_limit_ << "\n";
}

template <class CacheStore>
constexpr size_t GCCacheStore<CacheStore>::kMinCacheLimit;

// This class is the default cache state and store used by CacheBaseImpl.
// It uses VectorCacheStore for storage decorated by FirstCacheStore
// and GCCacheStore to do (optional) garbage collection.
template <class Arc>
class DefaultCacheStore
    : public GCCacheStore<FirstCacheStore<VectorCacheStore<CacheState<Arc>>>> {
 public:
  explicit DefaultCacheStore(const CacheOptions &opts)
      : GCCacheStore<FirstCacheStore<VectorCacheStore<CacheState<Arc>>>>(opts) {
  }
};

namespace internal {

// This class is used to cache FST elements stored in states of type State
// (see CacheState) with the flags used to indicate what has been cached. Use
// HasStart(), HasFinal(), and HasArcs() to determine if cached and SetStart(),
// SetFinal(), AddArc(), (or PushArc() and SetArcs()) to cache. Note that you
// must set the final weight even if the state is non-final to mark it as
// cached. The state storage method and any garbage collection policy are
// determined by the cache store. If the store is passed in with the options,
// CacheBaseImpl takes ownership.
template <class State,
          class CacheStore = DefaultCacheStore<typename State::Arc>>
class CacheBaseImpl : public FstImpl<typename State::Arc> {
 public:
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = CacheStore;

  using FstImpl<Arc>::Type;
  using FstImpl<Arc>::Properties;

  explicit CacheBaseImpl(const CacheOptions &opts = CacheOptions())
      : has_start_(false),
        cache_start_(kNoStateId),
        nknown_states_(0),
        min_unexpanded_state_id_(0),
        max_expanded_state_id_(-1),
        cache_gc_(opts.gc),
        cache_limit_(opts.gc_limit),
        cache_store_(new CacheStore(opts)),
        new_cache_store_(true),
        own_cache_store_(true) {}

  explicit CacheBaseImpl(const CacheImplOptions<CacheStore> &opts)
      : has_start_(false),
        cache_start_(kNoStateId),
        nknown_states_(0),
        min_unexpanded_state_id_(0),
        max_expanded_state_id_(-1),
        cache_gc_(opts.gc),
        cache_limit_(opts.gc_limit),
        cache_store_(opts.store ? opts.store : new CacheStore(CacheOptions(
                                                   opts.gc, opts.gc_limit))),
        new_cache_store_(!opts.store),
        own_cache_store_(opts.store ? opts.own_store : true) {}

  // Preserve gc parameters. If preserve_cache is true, also preserves
  // cache data.
  CacheBaseImpl(const CacheBaseImpl<State, CacheStore> &impl,
                bool preserve_cache = false)
      : FstImpl<Arc>(),
        has_start_(false),
        cache_start_(kNoStateId),
        nknown_states_(0),
        min_unexpanded_state_id_(0),
        max_expanded_state_id_(-1),
        cache_gc_(impl.cache_gc_),
        cache_limit_(impl.cache_limit_),
        cache_store_(new CacheStore(CacheOptions(cache_gc_, cache_limit_))),
        new_cache_store_(impl.new_cache_store_ || !preserve_cache),
        own_cache_store_(true) {
    if (preserve_cache) {
      *cache_store_ = *impl.cache_store_;
      has_start_ = impl.has_start_;
      cache_start_ = impl.cache_start_;
      nknown_states_ = impl.nknown_states_;
      expanded_states_ = impl.expanded_states_;
      min_unexpanded_state_id_ = impl.min_unexpanded_state_id_;
      max_expanded_state_id_ = impl.max_expanded_state_id_;
    }
  }

  ~CacheBaseImpl() override { if (own_cache_store_) delete cache_store_; }

  void SetStart(StateId s) {
    cache_start_ = s;
    has_start_ = true;
    if (s >= nknown_states_) nknown_states_ = s + 1;
  }

  void SetFinal(StateId s, Weight weight) {
    auto *state = cache_store_->GetMutableState(s);
    state->SetFinal(std::move(weight));
    static constexpr auto flags = kCacheFinal | kCacheRecent;
    state->SetFlags(flags, flags);
  }

// Disabled to ensure PushArc not AddArc is used in existing code
// TODO(sorenj): re-enable for backing store
#if 0
  // AddArc adds a single arc to a state and does incremental cache
  // book-keeping. For efficiency, prefer PushArc and SetArcs below
  // when possible.
  void AddArc(StateId s, const Arc &arc) {
    auto *state = cache_store_->GetMutableState(s);
    cache_store_->AddArc(state, arc);
    if (arc.nextstate >= nknown_states_)
      nknown_states_ = arc.nextstate + 1;
    SetExpandedState(s);
    static constexpr auto flags = kCacheArcs | kCacheRecent;
    state->SetFlags(flags, flags);
  }
#endif

  // Adds a single arc to a state but delays cache book-keeping. SetArcs must
  // be called when all PushArc calls at a state are complete. Do not mix with
  // calls to AddArc.
  void PushArc(StateId s, const Arc &arc) {
    auto *state = cache_store_->GetMutableState(s);
    state->PushArc(arc);
  }

  // Marks arcs of a state as cached and does cache book-keeping after all
  // calls to PushArc have been completed. Do not mix with calls to AddArc.
  void SetArcs(StateId s) {
    auto *state = cache_store_->GetMutableState(s);
    cache_store_->SetArcs(state);
    const auto narcs = state->NumArcs();
    for (size_t a = 0; a < narcs; ++a) {
      const auto &arc = state->GetArc(a);
      if (arc.nextstate >= nknown_states_) nknown_states_ = arc.nextstate + 1;
    }
    SetExpandedState(s);
    static constexpr auto flags = kCacheArcs | kCacheRecent;
    state->SetFlags(flags, flags);
  }

  void ReserveArcs(StateId s, size_t n) {
    auto *state = cache_store_->GetMutableState(s);
    state->ReserveArcs(n);
  }

  void DeleteArcs(StateId s) {
    auto *state = cache_store_->GetMutableState(s);
    cache_store_->DeleteArcs(state);
  }

  void DeleteArcs(StateId s, size_t n) {
    auto *state = cache_store_->GetMutableState(s);
    cache_store_->DeleteArcs(state, n);
  }

  void Clear() {
    nknown_states_ = 0;
    min_unexpanded_state_id_ = 0;
    max_expanded_state_id_ = -1;
    has_start_ = false;
    cache_start_ = kNoStateId;
    cache_store_->Clear();
  }

  // Is the start state cached?
  bool HasStart() const {
    if (!has_start_ && Properties(kError)) has_start_ = true;
    return has_start_;
  }

  // Is the final weight of the state cached?
  bool HasFinal(StateId s) const {
    const auto *state = cache_store_->GetState(s);
    if (state && state->Flags() & kCacheFinal) {
      state->SetFlags(kCacheRecent, kCacheRecent);
      return true;
    } else {
      return false;
    }
  }

  // Are arcs of the state cached?
  bool HasArcs(StateId s) const {
    const auto *state = cache_store_->GetState(s);
    if (state && state->Flags() & kCacheArcs) {
      state->SetFlags(kCacheRecent, kCacheRecent);
      return true;
    } else {
      return false;
    }
  }

  StateId Start() const { return cache_start_; }

  Weight Final(StateId s) const {
    const auto *state = cache_store_->GetState(s);
    return state->Final();
  }

  size_t NumArcs(StateId s) const {
    const auto *state = cache_store_->GetState(s);
    return state->NumArcs();
  }

  size_t NumInputEpsilons(StateId s) const {
    const auto *state = cache_store_->GetState(s);
    return state->NumInputEpsilons();
  }

  size_t NumOutputEpsilons(StateId s) const {
    const auto *state = cache_store_->GetState(s);
    return state->NumOutputEpsilons();
  }

  // Provides information needed for generic arc iterator.
  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const {
    const auto *state = cache_store_->GetState(s);
    data->base = nullptr;
    data->narcs = state->NumArcs();
    data->arcs = state->Arcs();
    data->ref_count = state->MutableRefCount();
    state->IncrRefCount();
  }

  // Number of known states.
  StateId NumKnownStates() const { return nknown_states_; }

  // Updates number of known states, taking into account the passed state ID.
  void UpdateNumKnownStates(StateId s) {
    if (s >= nknown_states_) nknown_states_ = s + 1;
  }

  // Finds the mininum never-expanded state ID.
  StateId MinUnexpandedState() const {
    while (min_unexpanded_state_id_ <= max_expanded_state_id_ &&
           ExpandedState(min_unexpanded_state_id_)) {
      ++min_unexpanded_state_id_;
    }
    return min_unexpanded_state_id_;
  }

  // Returns maximum ever-expanded state ID.
  StateId MaxExpandedState() const { return max_expanded_state_id_; }

  void SetExpandedState(StateId s) {
    if (s > max_expanded_state_id_) max_expanded_state_id_ = s;
    if (s < min_unexpanded_state_id_) return;
    if (s == min_unexpanded_state_id_) ++min_unexpanded_state_id_;
    if (cache_gc_ || cache_limit_ == 0) {
      if (expanded_states_.size() <= s) expanded_states_.resize(s + 1, false);
      expanded_states_[s] = true;
    }
  }

  bool ExpandedState(StateId s) const {
    if (cache_gc_ || cache_limit_ == 0) {
      return expanded_states_[s];
    } else if (new_cache_store_) {
      return cache_store_->GetState(s) != nullptr;
    } else {
      // If the cache was not created by this class, then the cached state needs
      // to be inspected to update nknown_states_.
      return false;
    }
  }

  const CacheStore *GetCacheStore() const { return cache_store_; }

  CacheStore *GetCacheStore() { return cache_store_; }

  // Caching on/off switch, limit and size accessors.

  bool GetCacheGc() const { return cache_gc_; }

  size_t GetCacheLimit() const { return cache_limit_; }

 private:
  mutable bool has_start_;                   // Is the start state cached?
  StateId cache_start_;                      // ID of start state.
  StateId nknown_states_;                    // Number of known states.
  std::vector<bool> expanded_states_;        // States that have been expanded.
  mutable StateId min_unexpanded_state_id_;  // Minimum never-expanded state ID
  mutable StateId max_expanded_state_id_;    // Maximum ever-expanded state ID
  bool cache_gc_;                            // GC enabled.
  size_t cache_limit_;       // Number of bytes allowed before GC.
  CacheStore *cache_store_;  // The store of cached states.
  bool new_cache_store_;     // Was the store was created by class?
  bool own_cache_store_;     // Is the store owned by class?

  CacheBaseImpl &operator=(const CacheBaseImpl &impl) = delete;
};

// A CacheBaseImpl with the default cache state type.
template <class Arc>
class CacheImpl : public CacheBaseImpl<CacheState<Arc>> {
 public:
  using State = CacheState<Arc>;

  CacheImpl() {}

  explicit CacheImpl(const CacheOptions &opts)
      : CacheBaseImpl<CacheState<Arc>>(opts) {}

  CacheImpl(const CacheImpl<Arc> &impl, bool preserve_cache = false)
      : CacheBaseImpl<State>(impl, preserve_cache) {}

 private:
  CacheImpl &operator=(const CacheImpl &impl) = delete;
};

}  // namespace internal

// Use this to make a state iterator for a CacheBaseImpl-derived FST, which must
// have Arc and Store types defined. Note this iterator only returns those
// states reachable from the initial state, so consider implementing a
// class-specific one.
//
// This class may be derived from.
template <class FST>
class CacheStateIterator : public StateIteratorBase<typename FST::Arc> {
 public:
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = typename FST::Store;
  using State = typename Store::State;
  using Impl = internal::CacheBaseImpl<State, Store>;

  CacheStateIterator(const FST &fst, Impl *impl)
      : fst_(fst), impl_(impl), s_(0) {
    fst_.Start();  // Forces start state.
  }

  bool Done() const final {
    if (s_ < impl_->NumKnownStates()) return false;
    for (StateId u = impl_->MinUnexpandedState(); u < impl_->NumKnownStates();
         u = impl_->MinUnexpandedState()) {
      // Forces state expansion.
      ArcIterator<FST> aiter(fst_, u);
      aiter.SetFlags(kArcValueFlags, kArcValueFlags | kArcNoCache);
      for (; !aiter.Done(); aiter.Next()) {
        impl_->UpdateNumKnownStates(aiter.Value().nextstate);
      }
      impl_->SetExpandedState(u);
      if (s_ < impl_->NumKnownStates()) return false;
    }
    return true;
  }

  StateId Value() const final { return s_; }

  void Next() final { ++s_; }

  void Reset() final { s_ = 0; }

 private:
  const FST &fst_;
  Impl *impl_;
  StateId s_;
};

// Used to make an arc iterator for a CacheBaseImpl-derived FST, which must
// have Arc and State types defined.
template <class FST>
class CacheArcIterator {
 public:
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = typename FST::Store;
  using State = typename Store::State;
  using Impl = internal::CacheBaseImpl<State, Store>;

  CacheArcIterator(Impl *impl, StateId s) : i_(0) {
    state_ = impl->GetCacheStore()->GetMutableState(s);
    state_->IncrRefCount();
  }

  ~CacheArcIterator() { state_->DecrRefCount(); }

  bool Done() const { return i_ >= state_->NumArcs(); }

  const Arc &Value() const { return state_->GetArc(i_); }

  void Next() { ++i_; }

  size_t Position() const { return i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  constexpr uint32 Flags() const { return kArcValueFlags; }

  void SetFlags(uint32 flags, uint32 mask) {}

 private:
  const State *state_;
  size_t i_;

  CacheArcIterator(const CacheArcIterator &) = delete;
  CacheArcIterator &operator=(const CacheArcIterator &) = delete;
};

// Use this to make a mutable arc iterator for a CacheBaseImpl-derived FST,
// which must have types Arc and Store defined.
template <class FST>
class CacheMutableArcIterator
    : public MutableArcIteratorBase<typename FST::Arc> {
 public:
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = typename FST::Store;
  using State = typename Store::State;
  using Impl = internal::CacheBaseImpl<State, Store>;

  // User must call MutateCheck() in the constructor.
  CacheMutableArcIterator(Impl *impl, StateId s) : i_(0), s_(s), impl_(impl) {
    state_ = impl_->GetCacheStore()->GetMutableState(s_);
    state_->IncrRefCount();
  }

  ~CacheMutableArcIterator() override { state_->DecrRefCount(); }

  bool Done() const final { return i_ >= state_->NumArcs(); }

  const Arc &Value() const final { return state_->GetArc(i_); }

  void Next() final { ++i_; }

  size_t Position() const final { return i_; }

  void Reset() final { i_ = 0; }

  void Seek(size_t a) final { i_ = a; }

  void SetValue(const Arc &arc) final { state_->SetArc(arc, i_); }

  uint32 Flags() const final { return kArcValueFlags; }

  void SetFlags(uint32, uint32) final {}

 private:
  size_t i_;
  StateId s_;
  Impl *impl_;
  State *state_;

  CacheMutableArcIterator(const CacheMutableArcIterator &) = delete;
  CacheMutableArcIterator &operator=(const CacheMutableArcIterator &) = delete;
};

// Wrap existing CacheStore implementation to use with ExpanderFst.
template <class CacheStore>
class ExpanderCacheStore {
 public:
  using State = typename CacheStore::State;
  using Arc = typename CacheStore::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit ExpanderCacheStore(const CacheOptions &opts = CacheOptions())
      : store_(opts) {}

  template <class Expander>
  State *FindOrExpand(Expander &expander, StateId s) {  // NOLINT
    auto *state = store_.GetMutableState(s);
    if (state->Flags()) {
      state->SetFlags(kCacheRecent, kCacheRecent);
    } else {
      StateBuilder builder(state);
      expander.Expand(s, &builder);
      state->SetFlags(kCacheFlags, kCacheFlags);
      store_.SetArcs(state);
    }
    return state;
  }

 private:
  CacheStore store_;

  struct StateBuilder {
    State *state;

    explicit StateBuilder(State *state_) : state(state_) {}

    void AddArc(const Arc &arc) { state->PushArc(arc); }

    void SetFinal(Weight weight) { state->SetFinal(std::move(weight)); }
  };
};

}  // namespace fst

#endif  // FST_CACHE_H_
