// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes to factor weights in an FST.

#ifndef FST_FACTOR_WEIGHT_H_
#define FST_FACTOR_WEIGHT_H_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/cache.h>
#include <fst/test-properties.h>


namespace fst {

constexpr uint32 kFactorFinalWeights = 0x00000001;
constexpr uint32 kFactorArcWeights = 0x00000002;

template <class Arc>
struct FactorWeightOptions : CacheOptions {
  using Label = typename Arc::Label;

  float delta;
  uint32 mode;         // Factor arc weights and/or final weights.
  Label final_ilabel;  // Input label of arc when factoring final weights.
  Label final_olabel;  // Output label of arc when factoring final weights.
  bool increment_final_ilabel;  // When factoring final w' results in > 1 arcs
  bool increment_final_olabel;  // at state, increment labels to make distinct?

  explicit FactorWeightOptions(const CacheOptions &opts, float delta = kDelta,
                               uint32 mode = kFactorArcWeights |
                                             kFactorFinalWeights,
                               Label final_ilabel = 0, Label final_olabel = 0,
                               bool increment_final_ilabel = false,
                               bool increment_final_olabel = false)
      : CacheOptions(opts),
        delta(delta),
        mode(mode),
        final_ilabel(final_ilabel),
        final_olabel(final_olabel),
        increment_final_ilabel(increment_final_ilabel),
        increment_final_olabel(increment_final_olabel) {}

  explicit FactorWeightOptions(float delta = kDelta,
                               uint32 mode = kFactorArcWeights |
                                             kFactorFinalWeights,
                               Label final_ilabel = 0, Label final_olabel = 0,
                               bool increment_final_ilabel = false,
                               bool increment_final_olabel = false)
      : delta(delta),
        mode(mode),
        final_ilabel(final_ilabel),
        final_olabel(final_olabel),
        increment_final_ilabel(increment_final_ilabel),
        increment_final_olabel(increment_final_olabel) {}
};

// A factor iterator takes as argument a weight w and returns a sequence of
// pairs of weights (xi, yi) such that the sum of the products xi times yi is
// equal to w. If w is fully factored, the iterator should return nothing.
//
// template <class W>
// class FactorIterator {
//  public:
//   explicit FactorIterator(W w);
//
//   bool Done() const;
//
//   void Next();
//
//   std::pair<W, W> Value() const;
//
//   void Reset();
// }

// Factors trivially.
template <class W>
class IdentityFactor {
 public:
  explicit IdentityFactor(const W &weight) {}

  bool Done() const { return true; }

  void Next() {}

  std::pair<W, W> Value() const { return std::make_pair(W::One(), W::One()); }

  void Reset() {}
};

// Factors a StringWeight w as 'ab' where 'a' is a label.
template <typename Label, StringType S = STRING_LEFT>
class StringFactor {
 public:
  explicit StringFactor(const StringWeight<Label, S> &weight)
      : weight_(weight), done_(weight.Size() <= 1) {}

  bool Done() const { return done_; }

  void Next() { done_ = true; }

  std::pair<StringWeight<Label, S>, StringWeight<Label, S>> Value() const {
    using Weight = StringWeight<Label, S>;
    typename Weight::Iterator siter(weight_);
    Weight w1(siter.Value());
    Weight w2;
    for (siter.Next(); !siter.Done(); siter.Next()) w2.PushBack(siter.Value());
    return std::make_pair(w1, w2);
  }

  void Reset() { done_ = weight_.Size() <= 1; }

 private:
  const StringWeight<Label, S> weight_;
  bool done_;
};

// Factor a GallicWeight using StringFactor.
template <class Label, class W, GallicType G = GALLIC_LEFT>
class GallicFactor {
 public:
  using GW = GallicWeight<Label, W, G>;

  explicit GallicFactor(const GW &weight)
      : weight_(weight), done_(weight.Value1().Size() <= 1) {}

  bool Done() const { return done_; }

  void Next() { done_ = true; }

  std::pair<GW, GW> Value() const {
    StringFactor<Label, GallicStringType(G)> siter(weight_.Value1());
    GW w1(siter.Value().first, weight_.Value2());
    GW w2(siter.Value().second, W::One());
    return std::make_pair(w1, w2);
  }

  void Reset() { done_ = weight_.Value1().Size() <= 1; }

 private:
  const GW weight_;
  bool done_;
};

// Specialization for the (general) GALLIC type GallicWeight.
template <class Label, class W>
class GallicFactor<Label, W, GALLIC> {
 public:
  using GW = GallicWeight<Label, W, GALLIC>;
  using GRW = GallicWeight<Label, W, GALLIC_RESTRICT>;

  explicit GallicFactor(const GW &weight)
      : iter_(weight),
        done_(weight.Size() == 0 ||
              (weight.Size() == 1 && weight.Back().Value1().Size() <= 1)) {}

  bool Done() const { return done_ || iter_.Done(); }

  void Next() { iter_.Next(); }

  void Reset() { iter_.Reset(); }

  std::pair<GW, GW> Value() const {
    const auto weight = iter_.Value();
    StringFactor<Label, GallicStringType(GALLIC_RESTRICT)> siter(
        weight.Value1());
    GRW w1(siter.Value().first, weight.Value2());
    GRW w2(siter.Value().second, W::One());
    return std::make_pair(GW(w1), GW(w2));
  }

 private:
  UnionWeightIterator<GRW, GallicUnionWeightOptions<Label, W>> iter_;
  bool done_;
};

namespace internal {

// Implementation class for FactorWeight
template <class Arc, class FactorIterator>
class FactorWeightFstImpl : public CacheImpl<Arc> {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  using CacheBaseImpl<CacheState<Arc>>::PushArc;
  using CacheBaseImpl<CacheState<Arc>>::HasStart;
  using CacheBaseImpl<CacheState<Arc>>::HasFinal;
  using CacheBaseImpl<CacheState<Arc>>::HasArcs;
  using CacheBaseImpl<CacheState<Arc>>::SetArcs;
  using CacheBaseImpl<CacheState<Arc>>::SetFinal;
  using CacheBaseImpl<CacheState<Arc>>::SetStart;

  struct Element {
    Element() {}

    Element(StateId s, Weight weight_) : state(s), weight(std::move(weight_)) {}

    StateId state;  // Input state ID.
    Weight weight;  // Residual weight.
  };

  FactorWeightFstImpl(const Fst<Arc> &fst, const FactorWeightOptions<Arc> &opts)
      : CacheImpl<Arc>(opts),
        fst_(fst.Copy()),
        delta_(opts.delta),
        mode_(opts.mode),
        final_ilabel_(opts.final_ilabel),
        final_olabel_(opts.final_olabel),
        increment_final_ilabel_(opts.increment_final_ilabel),
        increment_final_olabel_(opts.increment_final_olabel) {
    SetType("factor_weight");
    const auto props = fst.Properties(kFstProperties, false);
    SetProperties(FactorWeightProperties(props), kCopyProperties);
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
    if (mode_ == 0) {
      LOG(WARNING) << "FactorWeightFst: Factor mode is set to 0; "
                   << "factoring neither arc weights nor final weights";
    }
  }

  FactorWeightFstImpl(const FactorWeightFstImpl<Arc, FactorIterator> &impl)
      : CacheImpl<Arc>(impl),
        fst_(impl.fst_->Copy(true)),
        delta_(impl.delta_),
        mode_(impl.mode_),
        final_ilabel_(impl.final_ilabel_),
        final_olabel_(impl.final_olabel_),
        increment_final_ilabel_(impl.increment_final_ilabel_),
        increment_final_olabel_(impl.increment_final_olabel_) {
    SetType("factor_weight");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  StateId Start() {
    if (!HasStart()) {
      const auto s = fst_->Start();
      if (s == kNoStateId) return kNoStateId;
      SetStart(FindState(Element(fst_->Start(), Weight::One())));
    }
    return CacheImpl<Arc>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      const auto &element = elements_[s];
      // TODO(sorenj): fix so cast is unnecessary
      const auto weight =
          element.state == kNoStateId
              ? element.weight
              : (Weight)Times(element.weight, fst_->Final(element.state));
      FactorIterator siter(weight);
      if (!(mode_ & kFactorFinalWeights) || siter.Done()) {
        SetFinal(s, weight);
      } else {
        SetFinal(s, Weight::Zero());
      }
    }
    return CacheImpl<Arc>::Final(s);
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<Arc>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<Arc>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s)) Expand(s);
    return CacheImpl<Arc>::NumOutputEpsilons(s);
  }

  uint64 Properties() const override { return Properties(kFstProperties); }

  // Sets error if found, and returns other FST impl properties.
  uint64 Properties(uint64 mask) const override {
    if ((mask & kError) && fst_->Properties(kError, false)) {
      SetProperties(kError, kError);
    }
    return FstImpl<Arc>::Properties(mask);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
    if (!HasArcs(s)) Expand(s);
    CacheImpl<Arc>::InitArcIterator(s, data);
  }

  // Finds state corresponding to an element, creating new state if element not
  // found.
  StateId FindState(const Element &element) {
    if (!(mode_ & kFactorArcWeights) && element.weight == Weight::One() &&
        element.state != kNoStateId) {
      while (unfactored_.size() <= element.state)
        unfactored_.push_back(kNoStateId);
      if (unfactored_[element.state] == kNoStateId) {
        unfactored_[element.state] = elements_.size();
        elements_.push_back(element);
      }
      return unfactored_[element.state];
    } else {
      const auto insert_result =
          element_map_.insert(std::make_pair(element, elements_.size()));
      if (insert_result.second) {
        elements_.push_back(element);
      }
      return insert_result.first->second;
    }
  }

  // Computes the outgoing transitions from a state, creating new destination
  // states as needed.
  void Expand(StateId s) {
    const auto element = elements_[s];
    if (element.state != kNoStateId) {
      for (ArcIterator<Fst<Arc>> ait(*fst_, element.state); !ait.Done();
           ait.Next()) {
        const auto &arc = ait.Value();
        const auto weight = Times(element.weight, arc.weight);
        FactorIterator fiter(weight);
        if (!(mode_ & kFactorArcWeights) || fiter.Done()) {
          const auto dest = FindState(Element(arc.nextstate, Weight::One()));
          PushArc(s, Arc(arc.ilabel, arc.olabel, weight, dest));
        } else {
          for (; !fiter.Done(); fiter.Next()) {
            const auto &pair = fiter.Value();
            const auto dest =
                FindState(Element(arc.nextstate, pair.second.Quantize(delta_)));
            PushArc(s, Arc(arc.ilabel, arc.olabel, pair.first, dest));
          }
        }
      }
    }
    if ((mode_ & kFactorFinalWeights) &&
        ((element.state == kNoStateId) ||
         (fst_->Final(element.state) != Weight::Zero()))) {
      const auto weight =
          element.state == kNoStateId
              ? element.weight
              : Times(element.weight, fst_->Final(element.state));
      auto ilabel = final_ilabel_;
      auto olabel = final_olabel_;
      for (FactorIterator fiter(weight); !fiter.Done(); fiter.Next()) {
        const auto &pair = fiter.Value();
        const auto dest =
            FindState(Element(kNoStateId, pair.second.Quantize(delta_)));
        PushArc(s, Arc(ilabel, olabel, pair.first, dest));
        if (increment_final_ilabel_) ++ilabel;
        if (increment_final_olabel_) ++olabel;
      }
    }
    SetArcs(s);
  }

 private:
  // Equality function for Elements, assume weights have been quantized.
  class ElementEqual {
   public:
    bool operator()(const Element &x, const Element &y) const {
      return x.state == y.state && x.weight == y.weight;
    }
  };

  // Hash function for Elements to Fst states.
  class ElementKey {
   public:
    size_t operator()(const Element &x) const {
      static constexpr auto prime = 7853;
      return static_cast<size_t>(x.state * prime + x.weight.Hash());
    }
  };

  using ElementMap =
      std::unordered_map<Element, StateId, ElementKey, ElementEqual>;

  std::unique_ptr<const Fst<Arc>> fst_;
  float delta_;
  uint32 mode_;         // Factoring arc and/or final weights.
  Label final_ilabel_;  // ilabel of arc created when factoring final weights.
  Label final_olabel_;  // olabel of arc created when factoring final weights.
  bool increment_final_ilabel_;    // When factoring final weights results in
  bool increment_final_olabel_;    // mutiple arcs, increment labels?
  std::vector<Element> elements_;  // mapping from FST state to Element.
  ElementMap element_map_;         // mapping from Element to FST state.
  // Mapping between old/new StateId for states that do not need to be factored
  // when mode_ is 0 or kFactorFinalWeights.
  std::vector<StateId> unfactored_;
};

}  // namespace internal

// FactorWeightFst takes as template parameter a FactorIterator as defined
// above. The result of weight factoring is a transducer equivalent to the
// input whose path weights have been factored according to the FactorIterator.
// States and transitions will be added as necessary. The algorithm is a
// generalization to arbitrary weights of the second step of the input
// epsilon-normalization algorithm.
//
// This class attaches interface to implementation and handles reference
// counting, delegating most methods to ImplToFst.
template <class A, class FactorIterator>
class FactorWeightFst
    : public ImplToFst<internal::FactorWeightFstImpl<A, FactorIterator>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = DefaultCacheStore<Arc>;
  using State = typename Store::State;
  using Impl = internal::FactorWeightFstImpl<Arc, FactorIterator>;

  friend class ArcIterator<FactorWeightFst<Arc, FactorIterator>>;
  friend class StateIterator<FactorWeightFst<Arc, FactorIterator>>;

  explicit FactorWeightFst(const Fst<Arc> &fst)
      : ImplToFst<Impl>(
            std::make_shared<Impl>(fst, FactorWeightOptions<Arc>())) {}

  FactorWeightFst(const Fst<Arc> &fst, const FactorWeightOptions<Arc> &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, opts)) {}

  // See Fst<>::Copy() for doc.
  FactorWeightFst(const FactorWeightFst<Arc, FactorIterator> &fst, bool copy)
      : ImplToFst<Impl>(fst, copy) {}

  // Get a copy of this FactorWeightFst. See Fst<>::Copy() for further doc.
  FactorWeightFst<Arc, FactorIterator> *Copy(bool copy = false) const override {
    return new FactorWeightFst<Arc, FactorIterator>(*this, copy);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  FactorWeightFst &operator=(const FactorWeightFst &) = delete;
};

// Specialization for FactorWeightFst.
template <class Arc, class FactorIterator>
class StateIterator<FactorWeightFst<Arc, FactorIterator>>
    : public CacheStateIterator<FactorWeightFst<Arc, FactorIterator>> {
 public:
  explicit StateIterator(const FactorWeightFst<Arc, FactorIterator> &fst)
      : CacheStateIterator<FactorWeightFst<Arc, FactorIterator>>(
            fst, fst.GetMutableImpl()) {}
};

// Specialization for FactorWeightFst.
template <class Arc, class FactorIterator>
class ArcIterator<FactorWeightFst<Arc, FactorIterator>>
    : public CacheArcIterator<FactorWeightFst<Arc, FactorIterator>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const FactorWeightFst<Arc, FactorIterator> &fst, StateId s)
      : CacheArcIterator<FactorWeightFst<Arc, FactorIterator>>(
            fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class Arc, class FactorIterator>
inline void FactorWeightFst<Arc, FactorIterator>::InitStateIterator(
    StateIteratorData<Arc> *data) const {
  data->base = new StateIterator<FactorWeightFst<Arc, FactorIterator>>(*this);
}

}  // namespace fst

#endif  // FST_FACTOR_WEIGHT_H_
