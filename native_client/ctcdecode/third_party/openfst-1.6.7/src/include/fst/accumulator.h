// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes to accumulate arc weights. Useful for weight lookahead.

#ifndef FST_ACCUMULATOR_H_
#define FST_ACCUMULATOR_H_

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>

#include <fst/log.h>

#include <fst/arcfilter.h>
#include <fst/arcsort.h>
#include <fst/dfs-visit.h>
#include <fst/expanded-fst.h>
#include <fst/replace.h>

namespace fst {

// This class accumulates arc weights using the semiring Plus().
template <class A>
class DefaultAccumulator {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  DefaultAccumulator() {}

  DefaultAccumulator(const DefaultAccumulator &acc, bool safe = false) {}

  void Init(const Fst<Arc> &fst, bool copy = false) {}

  void SetState(StateId state) {}

  Weight Sum(Weight w, Weight v) { return Plus(w, v); }

  template <class ArcIter>
  Weight Sum(Weight w, ArcIter *aiter, ssize_t begin, ssize_t end) {
    Adder<Weight> adder(w);  // maintains cumulative sum accurately
    aiter->Seek(begin);
    for (auto pos = begin; pos < end; aiter->Next(), ++pos)
      adder.Add(aiter->Value().weight);
    return adder.Sum();
  }

  constexpr bool Error() const { return false; }

 private:
  DefaultAccumulator &operator=(const DefaultAccumulator &) = delete;
};

// This class accumulates arc weights using the log semiring Plus() assuming an
// arc weight has a WeightConvert specialization to and from log64 weights.
template <class A>
class LogAccumulator {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  LogAccumulator() {}

  LogAccumulator(const LogAccumulator &acc, bool safe = false) {}

  void Init(const Fst<Arc> &fst, bool copy = false) {}

  void SetState(StateId s) {}

  Weight Sum(Weight w, Weight v) { return LogPlus(w, v); }

  template <class ArcIter>
  Weight Sum(Weight w, ArcIter *aiter, ssize_t begin, ssize_t end) {
    auto sum = w;
    aiter->Seek(begin);
    for (auto pos = begin; pos < end; aiter->Next(), ++pos) {
      sum = LogPlus(sum, aiter->Value().weight);
    }
    return sum;
  }

  constexpr bool Error() const { return false; }

 private:
  Weight LogPlus(Weight w, Weight v) {
    if (w == Weight::Zero()) {
      return v;
    }
    const auto f1 = to_log_weight_(w).Value();
    const auto f2 = to_log_weight_(v).Value();
    if (f1 > f2) {
      return to_weight_(Log64Weight(f2 - internal::LogPosExp(f1 - f2)));
    } else {
      return to_weight_(Log64Weight(f1 - internal::LogPosExp(f2 - f1)));
    }
  }

  WeightConvert<Weight, Log64Weight> to_log_weight_;
  WeightConvert<Log64Weight, Weight> to_weight_;

  LogAccumulator &operator=(const LogAccumulator &) = delete;
};

// Interface for shareable data for fast log accumulator copies. Holds pointers
// to data only, storage is provided by derived classes.
class FastLogAccumulatorData {
 public:
  FastLogAccumulatorData(int arc_limit, int arc_period)
      : arc_limit_(arc_limit),
        arc_period_(arc_period),
        weights_ptr_(nullptr),
        num_weights_(0),
        weight_positions_ptr_(nullptr),
        num_positions_(0) {}

  virtual ~FastLogAccumulatorData() {}

  // Cummulative weight per state for all states s.t. # of arcs > arc_limit_
  // with arcs in order. The first element per state is Log64Weight::Zero().
  const double *Weights() const { return weights_ptr_; }

  int NumWeights() const { return num_weights_; }

  // Maps from state to corresponding beginning weight position in weights_.
  // osition -1 means no pre-computed weights for that state.
  const int *WeightPositions() const { return weight_positions_ptr_; }

  int NumPositions() const { return num_positions_; }

  int ArcLimit() const { return arc_limit_; }

  int ArcPeriod() const { return arc_period_; }

  // Returns true if the data object is mutable and supports SetData().
  virtual bool IsMutable() const = 0;

  // Does not take ownership but may invalidate the contents of weights and
  // weight_positions.
  virtual void SetData(std::vector<double> *weights,
                       std::vector<int> *weight_positions) = 0;

 protected:
  void Init(int num_weights, const double *weights, int num_positions,
            const int *weight_positions) {
    weights_ptr_ = weights;
    num_weights_ = num_weights;
    weight_positions_ptr_ = weight_positions;
    num_positions_ = num_positions;
  }

 private:
  const int arc_limit_;
  const int arc_period_;
  const double *weights_ptr_;
  int num_weights_;
  const int *weight_positions_ptr_;
  int num_positions_;

  FastLogAccumulatorData(const FastLogAccumulatorData &) = delete;
  FastLogAccumulatorData &operator=(const FastLogAccumulatorData &) = delete;
};

// FastLogAccumulatorData with mutable storage; filled by
// FastLogAccumulator::Init.
class MutableFastLogAccumulatorData : public FastLogAccumulatorData {
 public:
  MutableFastLogAccumulatorData(int arc_limit, int arc_period)
      : FastLogAccumulatorData(arc_limit, arc_period) {}

  bool IsMutable() const override { return true; }

  void SetData(std::vector<double> *weights,
               std::vector<int> *weight_positions) override {
    weights_.swap(*weights);
    weight_positions_.swap(*weight_positions);
    Init(weights_.size(), weights_.data(), weight_positions_.size(),
         weight_positions_.data());
  }

 private:
  std::vector<double> weights_;
  std::vector<int> weight_positions_;

  MutableFastLogAccumulatorData(const MutableFastLogAccumulatorData &) = delete;
  MutableFastLogAccumulatorData &operator=(
      const MutableFastLogAccumulatorData &) = delete;
};

// This class accumulates arc weights using the log semiring Plus() assuming an
// arc weight has a WeightConvert specialization to and from log64 weights. The
// member function Init(fst) has to be called to setup pre-computed weight
// information.
template <class A>
class FastLogAccumulator {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit FastLogAccumulator(ssize_t arc_limit = 20, ssize_t arc_period = 10)
      : to_log_weight_(),
        to_weight_(),
        arc_limit_(arc_limit),
        arc_period_(arc_period),
        data_(std::make_shared<MutableFastLogAccumulatorData>(arc_limit,
                                                              arc_period)),
        state_weights_(nullptr),
        error_(false) {}

  explicit FastLogAccumulator(std::shared_ptr<FastLogAccumulatorData> data)
      : to_log_weight_(),
        to_weight_(),
        arc_limit_(data->ArcLimit()),
        arc_period_(data->ArcPeriod()),
        data_(data),
        state_weights_(nullptr),
        error_(false) {}

  FastLogAccumulator(const FastLogAccumulator<Arc> &acc, bool safe = false)
      : to_log_weight_(),
        to_weight_(),
        arc_limit_(acc.arc_limit_),
        arc_period_(acc.arc_period_),
        data_(acc.data_),
        state_weights_(nullptr),
        error_(acc.error_) {}

  void SetState(StateId s) {
    const auto *weights = data_->Weights();
    const auto *weight_positions = data_->WeightPositions();
    state_weights_ = nullptr;
    if (s < data_->NumPositions()) {
      const auto pos = weight_positions[s];
      if (pos >= 0) state_weights_ = &(weights[pos]);
    }
  }

  Weight Sum(Weight w, Weight v) const { return LogPlus(w, v); }

  template <class ArcIter>
  Weight Sum(Weight w, ArcIter *aiter, ssize_t begin, ssize_t end) const {
    if (error_) return Weight::NoWeight();
    auto sum = w;
    // Finds begin and end of pre-stored weights.
    ssize_t index_begin = -1;
    ssize_t index_end = -1;
    ssize_t stored_begin = end;
    ssize_t stored_end = end;
    if (state_weights_) {
      index_begin = begin > 0 ? (begin - 1) / arc_period_ + 1 : 0;
      index_end = end / arc_period_;
      stored_begin = index_begin * arc_period_;
      stored_end = index_end * arc_period_;
    }
    // Computes sum before pre-stored weights.
    if (begin < stored_begin) {
      const auto pos_end = std::min(stored_begin, end);
      aiter->Seek(begin);
      for (auto pos = begin; pos < pos_end; aiter->Next(), ++pos) {
        sum = LogPlus(sum, aiter->Value().weight);
      }
    }
    // Computes sum between pre-stored weights.
    if (stored_begin < stored_end) {
      const auto f1 = state_weights_[index_end];
      const auto f2 = state_weights_[index_begin];
      if (f1 < f2) sum = LogPlus(sum, LogMinus(f1, f2));
      // Commented out for efficiency; adds Zero().
      /*
      else {
        // explicitly computes if cumulative sum lacks precision
        aiter->Seek(stored_begin);
        for (auto pos = stored_begin; pos < stored_end; aiter->Next(), ++pos)
          sum = LogPlus(sum, aiter->Value().weight);
      }
      */
    }
    // Computes sum after pre-stored weights.
    if (stored_end < end) {
      const auto pos_start = std::max(stored_begin, stored_end);
      aiter->Seek(pos_start);
      for (auto pos = pos_start; pos < end; aiter->Next(), ++pos) {
        sum = LogPlus(sum, aiter->Value().weight);
      }
    }
    return sum;
  }

  template <class FST>
  void Init(const FST &fst, bool copy = false) {
    if (copy || !data_->IsMutable()) return;
    if (data_->NumPositions() != 0 || arc_limit_ < arc_period_) {
      FSTERROR() << "FastLogAccumulator: Initialization error";
      error_ = true;
      return;
    }
    std::vector<double> weights;
    std::vector<int> weight_positions;
    weight_positions.reserve(CountStates(fst));
    for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
      const auto s = siter.Value();
      if (fst.NumArcs(s) >= arc_limit_) {
        auto sum = FloatLimits<double>::PosInfinity();
        if (weight_positions.size() <= s) weight_positions.resize(s + 1, -1);
        weight_positions[s] = weights.size();
        weights.push_back(sum);
        size_t narcs = 0;
        ArcIterator<FST> aiter(fst, s);
        aiter.SetFlags(kArcWeightValue | kArcNoCache, kArcFlags);
        for (; !aiter.Done(); aiter.Next()) {
          const auto &arc = aiter.Value();
          sum = LogPlus(sum, arc.weight);
          // Stores cumulative weight distribution per arc_period_.
          if (++narcs % arc_period_ == 0) weights.push_back(sum);
        }
      }
    }
    data_->SetData(&weights, &weight_positions);
  }

  bool Error() const { return error_; }

  std::shared_ptr<FastLogAccumulatorData> GetData() const { return data_; }

 private:
  static double LogPosExp(double x) {
    return x == FloatLimits<double>::PosInfinity() ? 0.0
                                                   : log(1.0F + exp(-x));
  }

  static double LogMinusExp(double x) {
    return x == FloatLimits<double>::PosInfinity() ? 0.0
                                                   : log(1.0F - exp(-x));
  }

  Weight LogPlus(Weight w, Weight v) const {
    if (w == Weight::Zero()) {
      return v;
    }
    const auto f1 = to_log_weight_(w).Value();
    const auto f2 = to_log_weight_(v).Value();
    if (f1 > f2) {
      return to_weight_(Log64Weight(f2 - LogPosExp(f1 - f2)));
    } else {
      return to_weight_(Log64Weight(f1 - LogPosExp(f2 - f1)));
    }
  }

  double LogPlus(double f1, Weight v) const {
    const auto f2 = to_log_weight_(v).Value();
    if (f1 == FloatLimits<double>::PosInfinity()) {
      return f2;
    } else if (f1 > f2) {
      return f2 - LogPosExp(f1 - f2);
    } else {
      return f1 - LogPosExp(f2 - f1);
    }
  }

  // Assumes f1 < f2.
  Weight LogMinus(double f1, double f2) const {
    if (f2 == FloatLimits<double>::PosInfinity()) {
      return to_weight_(Log64Weight(f1));
    } else {
      return to_weight_(Log64Weight(f1 - LogMinusExp(f2 - f1)));
    }
  }

  const WeightConvert<Weight, Log64Weight> to_log_weight_;
  const WeightConvert<Log64Weight, Weight> to_weight_;
  const ssize_t arc_limit_;   // Minimum number of arcs to pre-compute state.
  const ssize_t arc_period_;  // Saves cumulative weights per arc_period_.
  std::shared_ptr<FastLogAccumulatorData> data_;
  const double *state_weights_;
  bool error_;

  FastLogAccumulator &operator=(const FastLogAccumulator &) = delete;
};

// Stores shareable data for cache log accumulator copies. All copies share the
// same cache.
template <class Arc>
class CacheLogAccumulatorData {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  CacheLogAccumulatorData(bool gc, size_t gc_limit)
      : cache_gc_(gc), cache_limit_(gc_limit), cache_size_(0) {}

  CacheLogAccumulatorData(const CacheLogAccumulatorData<Arc> &data)
      : cache_gc_(data.cache_gc_),
        cache_limit_(data.cache_limit_),
        cache_size_(0) {}

  bool CacheDisabled() const { return cache_gc_ && cache_limit_ == 0; }

  std::vector<double> *GetWeights(StateId s) {
    auto it = cache_.find(s);
    if (it != cache_.end()) {
      it->second.recent = true;
      return it->second.weights.get();
    } else {
      return nullptr;
    }
  }

  void AddWeights(StateId s, std::vector<double> *weights) {
    if (cache_gc_ && cache_size_ >= cache_limit_) GC(false);
    cache_.insert(std::make_pair(s, CacheState(weights, true)));
    if (cache_gc_) cache_size_ += weights->capacity() * sizeof(double);
  }

 private:
  // Cached information for a given state.
  struct CacheState {
    std::unique_ptr<std::vector<double>> weights;  // Accumulated weights.
    bool recent;  // Has this state been accessed since last GC?

    CacheState(std::vector<double> *weights, bool recent)
        : weights(weights), recent(recent) {}
  };

  // Garbage collect: Deletes from cache states that have not been accessed
  // since the last GC ('free_recent = false') until 'cache_size_' is 2/3 of
  // 'cache_limit_'. If it does not free enough memory, start deleting
  // recently accessed states.
  void GC(bool free_recent) {
    auto cache_target = (2 * cache_limit_) / 3 + 1;
    auto it = cache_.begin();
    while (it != cache_.end() && cache_size_ > cache_target) {
      auto &cs = it->second;
      if (free_recent || !cs.recent) {
        cache_size_ -= cs.weights->capacity() * sizeof(double);
        cache_.erase(it++);
      } else {
        cs.recent = false;
        ++it;
      }
    }
    if (!free_recent && cache_size_ > cache_target) GC(true);
  }

  std::unordered_map<StateId, CacheState> cache_;  // Cache.
  bool cache_gc_;       // Enables garbage collection.
  size_t cache_limit_;  // # of bytes cached.
  size_t cache_size_;   // # of bytes allowed before GC.

  CacheLogAccumulatorData &operator=(const CacheLogAccumulatorData &) = delete;
};

// This class accumulates arc weights using the log semiring Plus() has a
// WeightConvert specialization to and from log64 weights. It is similar to the
// FastLogAccumator. However here, the accumulated weights are pre-computed and
// stored only for the states that are visited. The member function Init(fst)
// has to be called to setup this accumulator.
template <class Arc>
class CacheLogAccumulator {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit CacheLogAccumulator(ssize_t arc_limit = 10, bool gc = false,
                               size_t gc_limit = 10 * 1024 * 1024)
      : arc_limit_(arc_limit),
        data_(std::make_shared<CacheLogAccumulatorData<Arc>>(gc, gc_limit)),
        s_(kNoStateId),
        error_(false) {}

  CacheLogAccumulator(const CacheLogAccumulator<Arc> &acc, bool safe = false)
      : arc_limit_(acc.arc_limit_),
        fst_(acc.fst_ ? acc.fst_->Copy() : nullptr),
        data_(safe ? std::make_shared<CacheLogAccumulatorData<Arc>>(*acc.data_)
                   : acc.data_),
        s_(kNoStateId),
        error_(acc.error_) {}

  // Argument arc_limit specifies the minimum number of arcs to pre-compute.
  void Init(const Fst<Arc> &fst, bool copy = false) {
    if (!copy && fst_) {
      FSTERROR() << "CacheLogAccumulator: Initialization error";
      error_ = true;
      return;
    }
    fst_.reset(fst.Copy());
  }

  void SetState(StateId s, int depth = 0) {
    if (s == s_) return;
    s_ = s;
    if (data_->CacheDisabled() || error_) {
      weights_ = nullptr;
      return;
    }
    if (!fst_) {
      FSTERROR() << "CacheLogAccumulator::SetState: Incorrectly initialized";
      error_ = true;
      weights_ = nullptr;
      return;
    }
    weights_ = data_->GetWeights(s);
    if ((weights_ == nullptr) && (fst_->NumArcs(s) >= arc_limit_)) {
      weights_ = new std::vector<double>;
      weights_->reserve(fst_->NumArcs(s) + 1);
      weights_->push_back(FloatLimits<double>::PosInfinity());
      data_->AddWeights(s, weights_);
    }
  }

  Weight Sum(Weight w, Weight v) { return LogPlus(w, v); }

  template <class ArcIter>
  Weight Sum(Weight w, ArcIter *aiter, ssize_t begin, ssize_t end) {
    if (weights_ == nullptr) {
      auto sum = w;
      aiter->Seek(begin);
      for (auto pos = begin; pos < end; aiter->Next(), ++pos) {
        sum = LogPlus(sum, aiter->Value().weight);
      }
      return sum;
    } else {
      Extend(end, aiter);
      const auto &f1 = (*weights_)[end];
      const auto &f2 = (*weights_)[begin];
      if (f1 < f2) {
        return LogPlus(w, LogMinus(f1, f2));
      } else {
        // Commented out for efficiency; adds Zero().
        /*
        auto sum = w;
        // Explicitly computes if cumulative sum lacks precision.
        aiter->Seek(begin);
        for (auto pos = begin; pos < end; aiter->Next(), ++pos) {
          sum = LogPlus(sum, aiter->Value().weight);
        }
        return sum;
        */
        return w;
      }
    }
  }

  // Returns first position from aiter->Position() whose accumulated
  // value is greater or equal to w (w.r.t. Zero() < One()). The
  // iterator may be repositioned.
  template <class ArcIter>
  size_t LowerBound(Weight w, ArcIter *aiter) {
    const auto f = to_log_weight_(w).Value();
    auto pos = aiter->Position();
    if (weights_) {
      Extend(fst_->NumArcs(s_), aiter);
      return std::lower_bound(weights_->begin() + pos + 1, weights_->end(),
                              f, std::greater<double>()) -
          weights_->begin() - 1;
    } else {
      size_t n = 0;
      auto x = FloatLimits<double>::PosInfinity();
      for (aiter->Reset(); !aiter->Done(); aiter->Next(), ++n) {
        x = LogPlus(x, aiter->Value().weight);
        if (n >= pos && x <= f) break;
      }
      return n;
    }
  }

  bool Error() const { return error_; }

 private:
  double LogPosExp(double x) {
    return x == FloatLimits<double>::PosInfinity() ? 0.0
                                                   : log(1.0F + exp(-x));
  }

  double LogMinusExp(double x) {
    return x == FloatLimits<double>::PosInfinity() ? 0.0
                                                   : log(1.0F - exp(-x));
  }

  Weight LogPlus(Weight w, Weight v) {
    if (w == Weight::Zero()) {
      return v;
    }
    const auto f1 = to_log_weight_(w).Value();
    const auto f2 = to_log_weight_(v).Value();
    if (f1 > f2) {
      return to_weight_(Log64Weight(f2 - LogPosExp(f1 - f2)));
    } else {
      return to_weight_(Log64Weight(f1 - LogPosExp(f2 - f1)));
    }
  }

  double LogPlus(double f1, Weight v) {
    const auto f2 = to_log_weight_(v).Value();
    if (f1 == FloatLimits<double>::PosInfinity()) {
      return f2;
    } else if (f1 > f2) {
      return f2 - LogPosExp(f1 - f2);
    } else {
      return f1 - LogPosExp(f2 - f1);
    }
  }

  // Assumes f1 < f2.
  Weight LogMinus(double f1, double f2) {
    if (f2 == FloatLimits<double>::PosInfinity()) {
      return to_weight_(Log64Weight(f1));
    } else {
      return to_weight_(Log64Weight(f1 - LogMinusExp(f2 - f1)));
    }
  }

  // Extends weights up to index 'end'.
  template <class ArcIter>
  void Extend(ssize_t end, ArcIter *aiter) {
    if (weights_->size() <= end) {
      for (aiter->Seek(weights_->size() - 1); weights_->size() <= end;
           aiter->Next()) {
        weights_->push_back(LogPlus(weights_->back(), aiter->Value().weight));
      }
    }
  }


  WeightConvert<Weight, Log64Weight> to_log_weight_;
  WeightConvert<Log64Weight, Weight> to_weight_;
  ssize_t arc_limit_;                    // Minimum # of arcs to cache a state.
  std::vector<double> *weights_;         // Accumulated weights for cur. state.
  std::unique_ptr<const Fst<Arc>> fst_;  // Input FST.
  std::shared_ptr<CacheLogAccumulatorData<Arc>> data_;  // Cache data.
  StateId s_;                                           // Current state.
  bool error_;
};

// Stores shareable data for replace accumulator copies.
template <class Accumulator, class T>
class ReplaceAccumulatorData {
 public:
  using Arc = typename Accumulator::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using StateTable = T;
  using StateTuple = typename StateTable::StateTuple;

  ReplaceAccumulatorData() : state_table_(nullptr) {}

  explicit ReplaceAccumulatorData(
      const std::vector<Accumulator *> &accumulators)
      : state_table_(nullptr) {
    accumulators_.reserve(accumulators.size());
    for (const auto accumulator : accumulators) {
      accumulators_.emplace_back(accumulator);
    }
  }

  void Init(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_tuples,
            const StateTable *state_table) {
    state_table_ = state_table;
    accumulators_.resize(fst_tuples.size());
    for (Label i = 0; i < accumulators_.size(); ++i) {
      if (!accumulators_[i]) {
        accumulators_[i].reset(new Accumulator());
        accumulators_[i]->Init(*(fst_tuples[i].second));
      }
      fst_array_.emplace_back(fst_tuples[i].second->Copy());
    }
  }

  const StateTuple &GetTuple(StateId s) const { return state_table_->Tuple(s); }

  Accumulator *GetAccumulator(size_t i) { return accumulators_[i].get(); }

  const Fst<Arc> *GetFst(size_t i) const { return fst_array_[i].get(); }

 private:
  const StateTable *state_table_;
  std::vector<std::unique_ptr<Accumulator>> accumulators_;
  std::vector<std::unique_ptr<const Fst<Arc>>> fst_array_;
};

// This class accumulates weights in a ReplaceFst.  The 'Init' method takes as
// input the argument used to build the ReplaceFst and the ReplaceFst state
// table. It uses accumulators of type 'Accumulator' in the underlying FSTs.
template <class Accumulator,
          class T = DefaultReplaceStateTable<typename Accumulator::Arc>>
class ReplaceAccumulator {
 public:
  using Arc = typename Accumulator::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using StateTable = T;
  using StateTuple = typename StateTable::StateTuple;
  using Weight = typename Arc::Weight;

  ReplaceAccumulator()
      : init_(false),
        data_(std::make_shared<
              ReplaceAccumulatorData<Accumulator, StateTable>>()),
        error_(false) {}

  explicit ReplaceAccumulator(const std::vector<Accumulator *> &accumulators)
      : init_(false),
        data_(std::make_shared<ReplaceAccumulatorData<Accumulator, StateTable>>(
            accumulators)),
        error_(false) {}

  ReplaceAccumulator(const ReplaceAccumulator<Accumulator, StateTable> &acc,
                     bool safe = false)
      : init_(acc.init_), data_(acc.data_), error_(acc.error_) {
    if (!init_) {
      FSTERROR() << "ReplaceAccumulator: Can't copy unintialized accumulator";
    }
    if (safe) FSTERROR() << "ReplaceAccumulator: Safe copy not supported";
  }

  // Does not take ownership of the state table, the state table is owned by
  // the ReplaceFst.
  void Init(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_tuples,
            const StateTable *state_table) {
    init_ = true;
    data_->Init(fst_tuples, state_table);
  }

  // Method required by LookAheadMatcher. However, ReplaceAccumulator needs to
  // be initialized by calling the Init method above before being passed to
  // LookAheadMatcher.
  //
  // TODO(allauzen): Revisit this. Consider creating a method
  // Init(const ReplaceFst<A, T, C>&, bool) and using friendship to get access
  // to the innards of ReplaceFst.
  void Init(const Fst<Arc> &fst, bool copy = false) {
    if (!init_) {
      FSTERROR() << "ReplaceAccumulator::Init: Accumulator needs to be"
                 << " initialized before being passed to LookAheadMatcher";
      error_ = true;
    }
  }

  void SetState(StateId s) {
    if (!init_) {
      FSTERROR() << "ReplaceAccumulator::SetState: Incorrectly initialized";
      error_ = true;
      return;
    }
    auto tuple = data_->GetTuple(s);
    fst_id_ = tuple.fst_id - 1;  // Replace FST ID is 1-based.
    data_->GetAccumulator(fst_id_)->SetState(tuple.fst_state);
    if ((tuple.prefix_id != 0) &&
        (data_->GetFst(fst_id_)->Final(tuple.fst_state) != Weight::Zero())) {
      offset_ = 1;
      offset_weight_ = data_->GetFst(fst_id_)->Final(tuple.fst_state);
    } else {
      offset_ = 0;
      offset_weight_ = Weight::Zero();
    }
    aiter_.reset(
        new ArcIterator<Fst<Arc>>(*data_->GetFst(fst_id_), tuple.fst_state));
  }

  Weight Sum(Weight w, Weight v) {
    if (error_) return Weight::NoWeight();
    return data_->GetAccumulator(fst_id_)->Sum(w, v);
  }

  template <class ArcIter>
  Weight Sum(Weight w, ArcIter *aiter, ssize_t begin, ssize_t end) {
    if (error_) return Weight::NoWeight();
    auto sum = begin == end ? Weight::Zero()
                            : data_->GetAccumulator(fst_id_)->Sum(
                                  w, aiter_.get(), begin ? begin - offset_ : 0,
                                  end - offset_);
    if (begin == 0 && end != 0 && offset_ > 0) sum = Sum(offset_weight_, sum);
    return sum;
  }

  bool Error() const { return error_; }

 private:
  bool init_;
  std::shared_ptr<ReplaceAccumulatorData<Accumulator, StateTable>> data_;
  Label fst_id_;
  size_t offset_;
  Weight offset_weight_;
  std::unique_ptr<ArcIterator<Fst<Arc>>> aiter_;
  bool error_;
};

// SafeReplaceAccumulator accumulates weights in a ReplaceFst and copies of it
// are always thread-safe copies.
template <class Accumulator, class T>
class SafeReplaceAccumulator {
 public:
  using Arc = typename Accumulator::Arc;
  using StateId = typename Arc::StateId;
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;
  using StateTable = T;
  using StateTuple = typename StateTable::StateTuple;

  SafeReplaceAccumulator() {}

  SafeReplaceAccumulator(const SafeReplaceAccumulator &copy, bool safe)
      : SafeReplaceAccumulator(copy) {}

  explicit SafeReplaceAccumulator(
      const std::vector<Accumulator> &accumulators) {
    for (const auto &accumulator : accumulators) {
      accumulators_.emplace_back(accumulator, true);
    }
  }

  void Init(const std::vector<std::pair<Label, const Fst<Arc> *>> &fst_tuples,
            const StateTable *state_table) {
    state_table_ = state_table;
    for (Label i = 0; i < fst_tuples.size(); ++i) {
      if (i == accumulators_.size()) {
        accumulators_.resize(accumulators_.size() + 1);
        accumulators_[i].Init(*(fst_tuples[i].second));
      }
      fst_array_.emplace_back(fst_tuples[i].second->Copy(true));
    }
    init_ = true;
  }

  void Init(const Fst<Arc> &fst, bool copy = false) {
    if (!init_) {
      FSTERROR() << "SafeReplaceAccumulator::Init: Accumulator needs to be"
                 << " initialized before being passed to LookAheadMatcher";
      error_ = true;
    }
  }

  void SetState(StateId s) {
    auto tuple = state_table_->Tuple(s);
    fst_id_ = tuple.fst_id - 1;  // Replace FST ID is 1-based
    GetAccumulator(fst_id_)->SetState(tuple.fst_state);
    offset_ = 0;
    offset_weight_ = Weight::Zero();
    const auto final_weight = GetFst(fst_id_)->Final(tuple.fst_state);
    if ((tuple.prefix_id != 0) && (final_weight != Weight::Zero())) {
      offset_ = 1;
      offset_weight_ = final_weight;
    }
    aiter_.Set(*GetFst(fst_id_), tuple.fst_state);
  }

  Weight Sum(Weight w, Weight v) {
    if (error_) return Weight::NoWeight();
    return GetAccumulator(fst_id_)->Sum(w, v);
  }

  template <class ArcIter>
  Weight Sum(Weight w, ArcIter *aiter, ssize_t begin, ssize_t end) {
    if (error_) return Weight::NoWeight();
    if (begin == end) return Weight::Zero();
    auto sum = GetAccumulator(fst_id_)->Sum(
        w, aiter_.get(), begin ? begin - offset_ : 0, end - offset_);
    if (begin == 0 && end != 0 && offset_ > 0) {
      sum = Sum(offset_weight_, sum);
    }
    return sum;
  }

  bool Error() const { return error_; }

 private:
  class ArcIteratorPtr {
   public:
    ArcIteratorPtr() {}

    ArcIteratorPtr(const ArcIteratorPtr &copy) {}

    void Set(const Fst<Arc> &fst, StateId state_id) {
      ptr_.reset(new ArcIterator<Fst<Arc>>(fst, state_id));
    }

    ArcIterator<Fst<Arc>> *get() { return ptr_.get(); }

   private:
    std::unique_ptr<ArcIterator<Fst<Arc>>> ptr_;
  };

  Accumulator *GetAccumulator(size_t i) { return &accumulators_[i]; }

  const Fst<Arc> *GetFst(size_t i) const { return fst_array_[i].get(); }

  const StateTable *state_table_;
  std::vector<Accumulator> accumulators_;
  std::vector<std::shared_ptr<Fst<Arc>>> fst_array_;
  ArcIteratorPtr aiter_;
  bool init_ = false;
  bool error_ = false;
  Label fst_id_;
  size_t offset_;
  Weight offset_weight_;
};

}  // namespace fst

#endif  // FST_ACCUMULATOR_H_
