// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes and functions to generate random paths through an FST.

#ifndef FST_RANDGEN_H_
#define FST_RANDGEN_H_

#include <math.h>
#include <stddef.h>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/accumulator.h>
#include <fst/cache.h>
#include <fst/dfs-visit.h>
#include <fst/float-weight.h>
#include <fst/fst-decl.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/properties.h>
#include <fst/util.h>
#include <fst/weight.h>

namespace fst {

// The RandGenFst class is roughly similar to ArcMapFst in that it takes two
// template parameters denoting the input and output arc types. However, it also
// takes an additional template parameter which specifies a sampler object which
// samples (with replacement) arcs from an FST state. The sampler in turn takes
// a template parameter for a selector object which actually chooses the arc.
//
// Arc selector functors are used to select a random transition given an FST
// state s, returning a number N such that 0 <= N <= NumArcs(s). If N is
// NumArcs(s), then the final weight is selected; otherwise the N-th arc is
// selected. It is assumed these are not applied to any state which is neither
// final nor has any arcs leaving it.

// Randomly selects a transition using the uniform distribution. This class is
// not thread-safe.
template <class Arc>
class UniformArcSelector {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // Constructs a selector with a non-deterministic seed.
  UniformArcSelector() : rand_(std::random_device()()) {}
  // Constructs a selector with a given seed.
  explicit UniformArcSelector(uint64 seed) : rand_(seed) {}

  size_t operator()(const Fst<Arc> &fst, StateId s) const {
    const auto n = fst.NumArcs(s) + (fst.Final(s) != Weight::Zero());
    return static_cast<size_t>(
        std::uniform_int_distribution<>(0, n - 1)(rand_));
  }

 private:
  mutable std::mt19937_64 rand_;
};

// Randomly selects a transition w.r.t. the weights treated as negative log
// probabilities after normalizing for the total weight leaving the state. Zero
// transitions are disregarded. It assumed that Arc::Weight::Value() accesses
// the floating point representation of the weight. This class is not
// thread-safe.
template <class Arc>
class LogProbArcSelector {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // Constructs a selector with a non-deterministic seed.
  LogProbArcSelector() : rand_(std::random_device()()) {}
  // Constructs a selector with a given seed.
  explicit LogProbArcSelector(uint64 seed) : rand_(seed) {}

  size_t operator()(const Fst<Arc> &fst, StateId s) const {
    // Finds total weight leaving state.
    auto sum = Log64Weight::Zero();
    ArcIterator<Fst<Arc>> aiter(fst, s);
    for (; !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      sum = Plus(sum, to_log_weight_(arc.weight));
    }
    sum = Plus(sum, to_log_weight_(fst.Final(s)));
    const double threshold =
        std::uniform_real_distribution<>(0, exp(-sum.Value()))(rand_);
    auto p = Log64Weight::Zero();
    size_t n = 0;
    for (aiter.Reset(); !aiter.Done(); aiter.Next(), ++n) {
      p = Plus(p, to_log_weight_(aiter.Value().weight));
      if (exp(-p.Value()) > threshold) return n;
    }
    return n;
  }

 private:
  mutable std::mt19937_64 rand_;
  WeightConvert<Weight, Log64Weight> to_log_weight_;
};

// Useful alias when using StdArc.
using StdArcSelector = LogProbArcSelector<StdArc>;

// Same as LogProbArcSelector but use CacheLogAccumulator to cache the weight
// accumulation computations. This class is not thread-safe.
template <class Arc>
class FastLogProbArcSelector : public LogProbArcSelector<Arc> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using LogProbArcSelector<Arc>::operator();

  // Constructs a selector with a non-deterministic seed.
  FastLogProbArcSelector() : seed_(std::random_device()()), rand_(seed_) {}
  // Constructs a selector with a given seed.
  explicit FastLogProbArcSelector(uint64 seed) : seed_(seed), rand_(seed_) {}

  size_t operator()(const Fst<Arc> &fst, StateId s,
                    CacheLogAccumulator<Arc> *accumulator) const {
    accumulator->SetState(s);
    ArcIterator<Fst<Arc>> aiter(fst, s);
    // Finds total weight leaving state.
    const double sum = to_log_weight_(accumulator->Sum(fst.Final(s), &aiter, 0,
                                                       fst.NumArcs(s)))
                           .Value();
    const double r = -log(std::uniform_real_distribution<>(0, 1)(rand_));
    Weight w = from_log_weight_(r + sum);
    aiter.Reset();
    return accumulator->LowerBound(w, &aiter);
  }

  uint64 Seed() const { return seed_; }

 private:
  const uint64 seed_;
  mutable std::mt19937_64 rand_;
  WeightConvert<Weight, Log64Weight> to_log_weight_;
  WeightConvert<Log64Weight, Weight> from_log_weight_;
};

// Random path state info maintained by RandGenFst and passed to samplers.
template <typename Arc>
struct RandState {
  using StateId = typename Arc::StateId;

  StateId state_id;  // Current input FST state.
  size_t nsamples;   // Number of samples to be sampled at this state.
  size_t length;     // Length of path to this random state.
  size_t select;     // Previous sample arc selection.
  const RandState<Arc> *parent;  // Previous random state on this path.

  explicit RandState(StateId state_id, size_t nsamples = 0, size_t length = 0,
                     size_t select = 0, const RandState<Arc> *parent = nullptr)
      : state_id(state_id),
        nsamples(nsamples),
        length(length),
        select(select),
        parent(parent) {}

  RandState() : RandState(kNoStateId) {}
};

// This class, given an arc selector, samples, with replacement, multiple random
// transitions from an FST's state. This is a generic version with a
// straightforward use of the arc selector. Specializations may be defined for
// arc selectors for greater efficiency or special behavior.
template <class Arc, class Selector>
class ArcSampler {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // The max_length argument may be interpreted (or ignored) by a selector as
  // it chooses. This generic version interprets this literally.
  ArcSampler(const Fst<Arc> &fst, const Selector &selector,
             int32 max_length = std::numeric_limits<int32>::max())
      : fst_(fst), selector_(selector), max_length_(max_length) {}

  // Allow updating FST argument; pass only if changed.
  ArcSampler(const ArcSampler<Arc, Selector> &sampler,
             const Fst<Arc> *fst = nullptr)
      : fst_(fst ? *fst : sampler.fst_),
        selector_(sampler.selector_),
        max_length_(sampler.max_length_) {
    Reset();
  }

  // Samples a fixed number of samples from the given state. The length argument
  // specifies the length of the path to the state. Returns true if the samples
  // were collected. No samples may be collected if either there are no
  // transitions leaving the state and the state is non-final, or if the path
  // length has been exceeded. Iterator members are provided to read the samples
  // in the order in which they were collected.
  bool Sample(const RandState<Arc> &rstate) {
    sample_map_.clear();
    if ((fst_.NumArcs(rstate.state_id) == 0 &&
         fst_.Final(rstate.state_id) == Weight::Zero()) ||
        rstate.length == max_length_) {
      Reset();
      return false;
    }
    for (size_t i = 0; i < rstate.nsamples; ++i) {
      ++sample_map_[selector_(fst_, rstate.state_id)];
    }
    Reset();
    return true;
  }

  // More samples?
  bool Done() const { return sample_iter_ == sample_map_.end(); }

  // Gets the next sample.
  void Next() { ++sample_iter_; }

  std::pair<size_t, size_t> Value() const { return *sample_iter_; }

  void Reset() { sample_iter_ = sample_map_.begin(); }

  bool Error() const { return false; }

 private:
  const Fst<Arc> &fst_;
  const Selector &selector_;
  const int32 max_length_;

  // Stores (N, K) as described for Value().
  std::map<size_t, size_t> sample_map_;
  std::map<size_t, size_t>::const_iterator sample_iter_;

  ArcSampler<Arc, Selector> &operator=(const ArcSampler &) = delete;
};

// Samples one sample of num_to_sample dimensions from a multinomial
// distribution parameterized by a vector of probabilities. The result
// container should be pre-initialized (e.g., an empty map or a zeroed vector
// sized the same as the vector of probabilities.
// probs.size()).
template <class Result, class RNG>
void OneMultinomialSample(const std::vector<double> &probs,
                          size_t num_to_sample, Result *result, RNG *rng) {
  // Left-over probability mass.
  double norm = 0;
  for (double p : probs) norm += p;
  // Left-over number of samples needed.
  for (size_t i = 0; i < probs.size(); ++i) {
    size_t num_sampled = 0;
    if (probs[i] > 0) {
      std::binomial_distribution<> d(num_to_sample, probs[i] / norm);
      num_sampled = d(*rng);
    }
    if (num_sampled != 0) (*result)[i] = num_sampled;
    norm -= probs[i];
    num_to_sample -= num_sampled;
  }
}

// Specialization for FastLogProbArcSelector.
template <class Arc>
class ArcSampler<Arc, FastLogProbArcSelector<Arc>> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Accumulator = CacheLogAccumulator<Arc>;
  using Selector = FastLogProbArcSelector<Arc>;

  ArcSampler(const Fst<Arc> &fst, const Selector &selector,
             int32 max_length = std::numeric_limits<int32>::max())
      : fst_(fst),
        selector_(selector),
        max_length_(max_length),
        accumulator_(new Accumulator()) {
    accumulator_->Init(fst);
    rng_.seed(selector_.Seed());
  }

  ArcSampler(const ArcSampler<Arc, Selector> &sampler,
             const Fst<Arc> *fst = nullptr)
      : fst_(fst ? *fst : sampler.fst_),
        selector_(sampler.selector_),
        max_length_(sampler.max_length_) {
    if (fst) {
      accumulator_.reset(new Accumulator());
      accumulator_->Init(*fst);
    } else {  // Shallow copy.
      accumulator_.reset(new Accumulator(*sampler.accumulator_));
    }
  }

  bool Sample(const RandState<Arc> &rstate) {
    sample_map_.clear();
    if ((fst_.NumArcs(rstate.state_id) == 0 &&
         fst_.Final(rstate.state_id) == Weight::Zero()) ||
        rstate.length == max_length_) {
      Reset();
      return false;
    }
    if (fst_.NumArcs(rstate.state_id) + 1 < rstate.nsamples) {
      MultinomialSample(rstate);
      Reset();
      return true;
    }
    for (size_t i = 0; i < rstate.nsamples; ++i) {
      ++sample_map_[selector_(fst_, rstate.state_id, accumulator_.get())];
    }
    Reset();
    return true;
  }

  bool Done() const { return sample_iter_ == sample_map_.end(); }

  void Next() { ++sample_iter_; }

  std::pair<size_t, size_t> Value() const { return *sample_iter_; }

  void Reset() { sample_iter_ = sample_map_.begin(); }

  bool Error() const { return accumulator_->Error(); }

 private:
  using RNG = std::mt19937;

  // Sample according to the multinomial distribution of rstate.nsamples draws
  // from p_.
  void MultinomialSample(const RandState<Arc> &rstate) {
    p_.clear();
    for (ArcIterator<Fst<Arc>> aiter(fst_, rstate.state_id); !aiter.Done();
         aiter.Next()) {
      p_.push_back(exp(-to_log_weight_(aiter.Value().weight).Value()));
    }
    if (fst_.Final(rstate.state_id) != Weight::Zero()) {
      p_.push_back(exp(-to_log_weight_(fst_.Final(rstate.state_id)).Value()));
    }
    if (rstate.nsamples < std::numeric_limits<RNG::result_type>::max()) {
      OneMultinomialSample(p_, rstate.nsamples, &sample_map_, &rng_);
    } else {
      for (size_t i = 0; i < p_.size(); ++i) {
        sample_map_[i] = ceil(p_[i] * rstate.nsamples);
      }
    }
  }

  const Fst<Arc> &fst_;
  const Selector &selector_;
  const int32 max_length_;

  // Stores (N, K) for Value().
  std::map<size_t, size_t> sample_map_;
  std::map<size_t, size_t>::const_iterator sample_iter_;

  std::unique_ptr<Accumulator> accumulator_;
  RNG rng_;                // Random number generator.
  std::vector<double> p_;  // Multinomial parameters.
  WeightConvert<Weight, Log64Weight> to_log_weight_;
};

// Options for random path generation with RandGenFst. The template argument is
// a sampler, typically the class ArcSampler. Ownership of the sampler is taken
// by RandGenFst.
template <class Sampler>
struct RandGenFstOptions : public CacheOptions {
  Sampler *sampler;          // How to sample transitions at a state.
  int32 npath;               // Number of paths to generate.
  bool weighted;             // Is the output tree weighted by path count, or
                             // is it just an unweighted DAG?
  bool remove_total_weight;  // Remove total weight when output is weighted.

  RandGenFstOptions(const CacheOptions &opts, Sampler *sampler, int32 npath = 1,
                    bool weighted = true, bool remove_total_weight = false)
      : CacheOptions(opts),
        sampler(sampler),
        npath(npath),
        weighted(weighted),
        remove_total_weight(remove_total_weight) {}
};

namespace internal {

// Implementation of RandGenFst.
template <class FromArc, class ToArc, class Sampler>
class RandGenFstImpl : public CacheImpl<ToArc> {
 public:
  using FstImpl<ToArc>::SetType;
  using FstImpl<ToArc>::SetProperties;
  using FstImpl<ToArc>::SetInputSymbols;
  using FstImpl<ToArc>::SetOutputSymbols;

  using CacheBaseImpl<CacheState<ToArc>>::PushArc;
  using CacheBaseImpl<CacheState<ToArc>>::HasArcs;
  using CacheBaseImpl<CacheState<ToArc>>::HasFinal;
  using CacheBaseImpl<CacheState<ToArc>>::HasStart;
  using CacheBaseImpl<CacheState<ToArc>>::SetArcs;
  using CacheBaseImpl<CacheState<ToArc>>::SetFinal;
  using CacheBaseImpl<CacheState<ToArc>>::SetStart;

  using Label = typename FromArc::Label;
  using StateId = typename FromArc::StateId;
  using FromWeight = typename FromArc::Weight;

  using ToWeight = typename ToArc::Weight;

  RandGenFstImpl(const Fst<FromArc> &fst,
                 const RandGenFstOptions<Sampler> &opts)
      : CacheImpl<ToArc>(opts),
        fst_(fst.Copy()),
        sampler_(opts.sampler),
        npath_(opts.npath),
        weighted_(opts.weighted),
        remove_total_weight_(opts.remove_total_weight),
        superfinal_(kNoLabel) {
    SetType("randgen");
    SetProperties(
        RandGenProperties(fst.Properties(kFstProperties, false), weighted_),
        kCopyProperties);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  RandGenFstImpl(const RandGenFstImpl &impl)
      : CacheImpl<ToArc>(impl),
        fst_(impl.fst_->Copy(true)),
        sampler_(new Sampler(*impl.sampler_, fst_.get())),
        npath_(impl.npath_),
        weighted_(impl.weighted_),
        superfinal_(kNoLabel) {
    SetType("randgen");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  StateId Start() {
    if (!HasStart()) {
      const auto s = fst_->Start();
      if (s == kNoStateId) return kNoStateId;
      SetStart(state_table_.size());
      state_table_.emplace_back(
          new RandState<FromArc>(s, npath_, 0, 0, nullptr));
    }
    return CacheImpl<ToArc>::Start();
  }

  ToWeight Final(StateId s) {
    if (!HasFinal(s)) Expand(s);
    return CacheImpl<ToArc>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<ToArc>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<ToArc>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<ToArc>::NumOutputEpsilons(s);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) &&
        (fst_->Properties(kError, false) || sampler_->Error())) {
      SetProperties(kError, kError);
    }
    return FstImpl<ToArc>::Properties(mask);
  }

  void InitArcIterator(StateId s, ArcIteratorData<ToArc> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl<ToArc>::InitArcIterator(s, data);
  }

  // Computes the outgoing transitions from a state, creating new destination
  // states as needed.
  void Expand(StateId s) {
    if (s == superfinal_) {
      SetFinal(s, ToWeight::One());
      SetArcs(s);
      return;
    }
    SetFinal(s, ToWeight::Zero());
    const auto &rstate = *state_table_[s];
    sampler_->Sample(rstate);
    ArcIterator<Fst<FromArc>> aiter(*fst_, rstate.state_id);
    const auto narcs = fst_->NumArcs(rstate.state_id);
    for (; !sampler_->Done(); sampler_->Next()) {
      const auto &sample_pair = sampler_->Value();
      const auto pos = sample_pair.first;
      const auto count = sample_pair.second;
      double prob = static_cast<double>(count) / rstate.nsamples;
      if (pos < narcs) {  // Regular transition.
        aiter.Seek(sample_pair.first);
        const auto &aarc = aiter.Value();
        const auto weight =
            weighted_ ? to_weight_(Log64Weight(-log(prob))) : ToWeight::One();
        const ToArc barc(aarc.ilabel, aarc.olabel, weight, state_table_.size());
        PushArc(s, barc);
        auto *nrstate = new RandState<FromArc>(aarc.nextstate, count,
                                               rstate.length + 1, pos, &rstate);
        state_table_.emplace_back(nrstate);
      } else {  // Super-final transition.
        if (weighted_) {
          const auto weight =
              remove_total_weight_
                  ? to_weight_(Log64Weight(-log(prob)))
                  : to_weight_(Log64Weight(-log(prob * npath_)));
          SetFinal(s, weight);
        } else {
          if (superfinal_ == kNoLabel) {
            superfinal_ = state_table_.size();
            state_table_.emplace_back(
                new RandState<FromArc>(kNoStateId, 0, 0, 0, nullptr));
          }
          for (size_t n = 0; n < count; ++n) {
            const ToArc barc(0, 0, ToWeight::One(), superfinal_);
            PushArc(s, barc);
          }
        }
      }
    }
    SetArcs(s);
  }

 private:
  const std::unique_ptr<Fst<FromArc>> fst_;
  std::unique_ptr<Sampler> sampler_;
  const int32 npath_;
  std::vector<std::unique_ptr<RandState<FromArc>>> state_table_;
  const bool weighted_;
  bool remove_total_weight_;
  StateId superfinal_;
  WeightConvert<Log64Weight, ToWeight> to_weight_;
};

}  // namespace internal

// FST class to randomly generate paths through an FST, with details controlled
// by RandGenOptionsFst. Output format is a tree weighted by the path count.
template <class FromArc, class ToArc, class Sampler>
class RandGenFst
    : public ImplToFst<internal::RandGenFstImpl<FromArc, ToArc, Sampler>> {
 public:
  using Label = typename FromArc::Label;
  using StateId = typename FromArc::StateId;
  using Weight = typename FromArc::Weight;

  using Store = DefaultCacheStore<FromArc>;
  using State = typename Store::State;

  using Impl = internal::RandGenFstImpl<FromArc, ToArc, Sampler>;

  friend class ArcIterator<RandGenFst<FromArc, ToArc, Sampler>>;
  friend class StateIterator<RandGenFst<FromArc, ToArc, Sampler>>;

  RandGenFst(const Fst<FromArc> &fst, const RandGenFstOptions<Sampler> &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, opts)) {}

  // See Fst<>::Copy() for doc.
  RandGenFst(const RandGenFst<FromArc, ToArc, Sampler> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  // Get a copy of this RandGenFst. See Fst<>::Copy() for further doc.
  RandGenFst<FromArc, ToArc, Sampler> *Copy(bool safe = false) const override {
    return new RandGenFst<FromArc, ToArc, Sampler>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<ToArc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<ToArc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  RandGenFst &operator=(const RandGenFst &) = delete;
};

// Specialization for RandGenFst.
template <class FromArc, class ToArc, class Sampler>
class StateIterator<RandGenFst<FromArc, ToArc, Sampler>>
    : public CacheStateIterator<RandGenFst<FromArc, ToArc, Sampler>> {
 public:
  explicit StateIterator(const RandGenFst<FromArc, ToArc, Sampler> &fst)
      : CacheStateIterator<RandGenFst<FromArc, ToArc, Sampler>>(
            fst, fst.GetMutableImpl()) {}
};

// Specialization for RandGenFst.
template <class FromArc, class ToArc, class Sampler>
class ArcIterator<RandGenFst<FromArc, ToArc, Sampler>>
    : public CacheArcIterator<RandGenFst<FromArc, ToArc, Sampler>> {
 public:
  using StateId = typename FromArc::StateId;

  ArcIterator(const RandGenFst<FromArc, ToArc, Sampler> &fst, StateId s)
      : CacheArcIterator<RandGenFst<FromArc, ToArc, Sampler>>(
            fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class FromArc, class ToArc, class Sampler>
inline void RandGenFst<FromArc, ToArc, Sampler>::InitStateIterator(
    StateIteratorData<ToArc> *data) const {
  data->base = new StateIterator<RandGenFst<FromArc, ToArc, Sampler>>(*this);
}

// Options for random path generation.
template <class Selector>
struct RandGenOptions {
  const Selector &selector;  // How an arc is selected at a state.
  int32 max_length;          // Maximum path length.
  int32 npath;               // Number of paths to generate.
  bool weighted;             // Is the output tree weighted by path count, or
                             // is it just an unweighted DAG?
  bool remove_total_weight;  // Remove total weight when output is weighted?

  explicit RandGenOptions(const Selector &selector,
                          int32 max_length = std::numeric_limits<int32>::max(),
                          int32 npath = 1, bool weighted = false,
                          bool remove_total_weight = false)
      : selector(selector),
        max_length(max_length),
        npath(npath),
        weighted(weighted),
        remove_total_weight(remove_total_weight) {}
};

namespace internal {

template <class FromArc, class ToArc>
class RandGenVisitor {
 public:
  using StateId = typename FromArc::StateId;
  using Weight = typename FromArc::Weight;

  explicit RandGenVisitor(MutableFst<ToArc> *ofst) : ofst_(ofst) {}

  void InitVisit(const Fst<FromArc> &ifst) {
    ifst_ = &ifst;
    ofst_->DeleteStates();
    ofst_->SetInputSymbols(ifst.InputSymbols());
    ofst_->SetOutputSymbols(ifst.OutputSymbols());
    if (ifst.Properties(kError, false)) ofst_->SetProperties(kError, kError);
    path_.clear();
  }

  constexpr bool InitState(StateId, StateId) const { return true; }

  bool TreeArc(StateId, const ToArc &arc) {
    if (ifst_->Final(arc.nextstate) == Weight::Zero()) {
      path_.push_back(arc);
    } else {
      OutputPath();
    }
    return true;
  }

  bool BackArc(StateId, const FromArc &) {
    FSTERROR() << "RandGenVisitor: cyclic input";
    ofst_->SetProperties(kError, kError);
    return false;
  }

  bool ForwardOrCrossArc(StateId, const FromArc &) {
    OutputPath();
    return true;
  }

  void FinishState(StateId s, StateId p, const FromArc *) {
    if (p != kNoStateId && ifst_->Final(s) == Weight::Zero()) path_.pop_back();
  }

  void FinishVisit() {}

 private:
  void OutputPath() {
    if (ofst_->Start() == kNoStateId) {
      const auto start = ofst_->AddState();
      ofst_->SetStart(start);
    }
    auto src = ofst_->Start();
    for (size_t i = 0; i < path_.size(); ++i) {
      const auto dest = ofst_->AddState();
      const ToArc arc(path_[i].ilabel, path_[i].olabel, Weight::One(), dest);
      ofst_->AddArc(src, arc);
      src = dest;
    }
    ofst_->SetFinal(src, Weight::One());
  }

  const Fst<FromArc> *ifst_;
  MutableFst<ToArc> *ofst_;
  std::vector<ToArc> path_;

  RandGenVisitor(const RandGenVisitor &) = delete;
  RandGenVisitor &operator=(const RandGenVisitor &) = delete;
};

}  // namespace internal

// Randomly generate paths through an FST; details controlled by
// RandGenOptions.
template <class FromArc, class ToArc, class Selector>
void RandGen(const Fst<FromArc> &ifst, MutableFst<ToArc> *ofst,
             const RandGenOptions<Selector> &opts) {
  using State = typename ToArc::StateId;
  using Weight = typename ToArc::Weight;
  using Sampler = ArcSampler<FromArc, Selector>;
  auto *sampler = new Sampler(ifst, opts.selector, opts.max_length);
  RandGenFstOptions<Sampler> fopts(CacheOptions(true, 0), sampler, opts.npath,
                                   opts.weighted, opts.remove_total_weight);
  RandGenFst<FromArc, ToArc, Sampler> rfst(ifst, fopts);
  if (opts.weighted) {
    *ofst = rfst;
  } else {
    internal::RandGenVisitor<FromArc, ToArc> rand_visitor(ofst);
    DfsVisit(rfst, &rand_visitor);
  }
}

// Randomly generate a path through an FST with the uniform distribution
// over the transitions.
template <class FromArc, class ToArc>
void RandGen(const Fst<FromArc> &ifst, MutableFst<ToArc> *ofst) {
  const UniformArcSelector<FromArc> uniform_selector;
  RandGenOptions<UniformArcSelector<ToArc>> opts(uniform_selector);
  RandGen(ifst, ofst, opts);
}

}  // namespace fst

#endif  // FST_RANDGEN_H_
