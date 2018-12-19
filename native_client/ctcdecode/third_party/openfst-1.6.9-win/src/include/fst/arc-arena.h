#ifndef FST_ARC_ARENA_H_
#define FST_ARC_ARENA_H_

#include <deque>
#include <memory>
#include <utility>
#include <fst/fst.h>
#include <fst/memory.h>
#include <unordered_map>

namespace fst {

// ArcArena is used for fast allocation of contiguous arrays of arcs.
//
// To create an arc array:
//   for each state:
//     for each arc:
//       arena.PushArc();
//     // Commits these arcs and returns pointer to them.
//     Arc *arcs = arena.GetArcs();
//
//     OR
//
//     arena.DropArcs();  // Throws away current arcs, reuse the space.
//
// The arcs returned are guaranteed to be contiguous and the pointer returned
// will never be invalidated until the arena is cleared for reuse.
//
// The contents of the arena can be released with a call to arena.Clear() after
// which the arena will restart with an initial allocation capable of holding at
// least all of the arcs requested in the last usage before Clear() making
// subsequent uses of the Arena more efficient.
//
// The max_retained_size option can limit the amount of arc space requested on
// Clear() to avoid excess growth from intermittent high usage.
template <typename Arc>
class ArcArena {
 public:
  explicit ArcArena(size_t block_size = 256,
                    size_t max_retained_size = 1e6)
      : block_size_(block_size),
        max_retained_size_(max_retained_size) {
    blocks_.emplace_back(MakeSharedBlock(block_size_));
    first_block_size_ = block_size_;
    total_size_ = block_size_;
    arcs_ = blocks_.back().get();
    end_ = arcs_ + block_size_;
    next_ = arcs_;
  }

  ArcArena(const ArcArena& copy)
      : arcs_(copy.arcs_), next_(copy.next_), end_(copy.end_),
        block_size_(copy.block_size_),
        first_block_size_(copy.first_block_size_),
        total_size_(copy.total_size_),
        max_retained_size_(copy.max_retained_size_),
        blocks_(copy.blocks_) {
    NewBlock(block_size_);
  }

  void ReserveArcs(size_t n) {
    if (next_ + n < end_) return;
    NewBlock(n);
  }

  void PushArc(const Arc& arc) {
    if (next_ == end_) {
      size_t length = next_ - arcs_;
      NewBlock(length * 2);
    }
    *next_ = arc;
    ++next_;
  }

  const Arc* GetArcs() {
    const auto *arcs = arcs_;
    arcs_ = next_;
    return arcs;
  }

  void DropArcs() { next_ = arcs_; }

  size_t Size() { return total_size_; }

  void Clear() {
    blocks_.resize(1);
    if (total_size_ > first_block_size_) {
      first_block_size_ = std::min(max_retained_size_, total_size_);
      blocks_.back() = MakeSharedBlock(first_block_size_);
    }
    total_size_ = first_block_size_;
    arcs_ = blocks_.back().get();
    end_ = arcs_ + first_block_size_;
    next_ = arcs_;
  }

 private:
  // Allocates a new block with capacity of at least n or block_size,
  // copying incomplete arc sequence from old block to new block.
  void NewBlock(size_t n) {
    const auto length = next_ - arcs_;
    const auto new_block_size = std::max(n, block_size_);
    total_size_ += new_block_size;
    blocks_.emplace_back(MakeSharedBlock(new_block_size));
    std::copy(arcs_, next_, blocks_.back().get());
    arcs_ = blocks_.back().get();
    next_ = arcs_ + length;
    end_ = arcs_ + new_block_size;
  }

  std::shared_ptr<Arc> MakeSharedBlock(size_t size) {
    return std::shared_ptr<Arc>(new Arc[size], std::default_delete<Arc[]>());
  }

  Arc *arcs_;
  Arc *next_;
  const Arc *end_;
  size_t block_size_;
  size_t first_block_size_;
  size_t total_size_;
  size_t max_retained_size_;
  std::list<std::shared_ptr<Arc>> blocks_;
};

// ArcArenaStateStore uses a resusable ArcArena to store arc arrays and does not
// require that the Expander call ReserveArcs first.
//
// TODO(tombagby): Make cache type configurable.
// TODO(tombagby): Provide ThreadLocal/Concurrent configuration.
template <class A>
class ArcArenaStateStore {
 public:
  using Arc = A;
  using Weight = typename Arc::Weight;
  using StateId = typename Arc::StateId;

  ArcArenaStateStore() : arena_(64 * 1024) {
  }

  class State {
   public:
    Weight Final() const { return final_; }

    size_t NumInputEpsilons() const { return niepsilons_; }

    size_t NumOutputEpsilons() const { return noepsilons_; }

    size_t NumArcs() const { return narcs_; }

    const Arc &GetArc(size_t n) const { return arcs_[n]; }

    const Arc *Arcs() const { return arcs_; }

    int* MutableRefCount() const { return nullptr; }

   private:
    State(Weight weight, int32_t niepsilons, int32_t noepsilons, int32_t narcs,
          const Arc *arcs)
        : final_(std::move(weight)),
          niepsilons_(niepsilons),
          noepsilons_(noepsilons),
          narcs_(narcs),
          arcs_(arcs) {}

    Weight final_;
    size_t niepsilons_;
    size_t noepsilons_;
    size_t narcs_;
    const Arc *arcs_;

    friend class ArcArenaStateStore<Arc>;
  };

  template <class Expander>
  State *FindOrExpand(Expander &expander, StateId state_id) {  // NOLINT
    auto it = cache_.insert(std::pair<StateId, State*>(state_id, nullptr));
    if (!it.second) return it.first->second;
    // Needs a new state.
    StateBuilder builder(&arena_);
    expander.Expand(state_id, &builder);
    const auto arcs = arena_.GetArcs();
    size_t narcs = builder.narcs_;
    size_t niepsilons = 0;
    size_t noepsilons = 0;
    for (size_t i = 0; i < narcs; ++i) {
      if (arcs[i].ilabel == 0) ++niepsilons;
      if (arcs[i].olabel == 0) ++noepsilons;
    }
    states_.emplace_back(
        State(builder.final_, niepsilons, noepsilons, narcs, arcs));
    // Places it in the cache.
    auto state = &states_.back();
    it.first->second = state;
    return state;
  }

  State *Find(StateId state_id) const {
    auto it = cache_.find(state_id);
    return (it == cache_.end()) ? nullptr : it->second;
  }

 private:
  class StateBuilder {
   public:
    explicit StateBuilder(ArcArena<Arc>* arena)
       : arena_(arena), final_(Weight::Zero()), narcs_(0) {}

    void SetFinal(Weight weight) { final_ = std::move(weight); }

    void ReserveArcs(size_t n) { arena_->ReserveArcs(n); }

    void AddArc(const Arc &arc) {
      ++narcs_;
      arena_->PushArc(arc);
    }

   private:
    friend class ArcArenaStateStore<Arc>;

    ArcArena<Arc> *arena_;
    Weight final_;
    size_t narcs_;
  };

  std::unordered_map<StateId, State *> cache_;
  std::deque<State> states_;
  ArcArena<Arc> arena_;
};

}  // namespace fst

#endif  // FST_ARC_ARENA_H_
