// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Simple concrete, mutable FST whose states and arcs are stored in STL vectors.

#ifndef FST_VECTOR_FST_H_
#define FST_VECTOR_FST_H_

#include <string>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/fst-decl.h>  // For optional argument declarations
#include <fst/mutable-fst.h>
#include <fst/test-properties.h>


namespace fst {

template <class A, class S>
class VectorFst;

template <class F, class G>
void Cast(const F &, G *);

// Arcs (of type A) implemented by an STL vector per state. M specifies Arc
// allocator (default declared in fst-decl.h).
template <class A, class M /* = std::allocator<A> */>
class VectorState {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using ArcAllocator = M;
  using StateAllocator =
      typename ArcAllocator::template rebind<VectorState<Arc, M>>::other;

  // Provide STL allocator for arcs.
  explicit VectorState(const ArcAllocator &alloc)
      : final_(Weight::Zero()), niepsilons_(0), noepsilons_(0), arcs_(alloc) {}

  VectorState(const VectorState<A, M> &state, const ArcAllocator &alloc)
      : final_(state.Final()),
        niepsilons_(state.NumInputEpsilons()),
        noepsilons_(state.NumOutputEpsilons()),
        arcs_(state.arcs_.begin(), state.arcs_.end(), alloc) {}

  void Reset() {
    final_ = Weight::Zero();
    niepsilons_ = 0;
    noepsilons_ = 0;
    arcs_.clear();
  }

  Weight Final() const { return final_; }

  size_t NumInputEpsilons() const { return niepsilons_; }

  size_t NumOutputEpsilons() const { return noepsilons_; }

  size_t NumArcs() const { return arcs_.size(); }

  const Arc &GetArc(size_t n) const { return arcs_[n]; }

  const Arc *Arcs() const { return !arcs_.empty() ? &arcs_[0] : nullptr; }

  Arc *MutableArcs() { return !arcs_.empty() ? &arcs_[0] : nullptr; }

  void ReserveArcs(size_t n) { arcs_.reserve(n); }

  void SetFinal(Weight weight) { final_ = std::move(weight); }

  void SetNumInputEpsilons(size_t n) { niepsilons_ = n; }

  void SetNumOutputEpsilons(size_t n) { noepsilons_ = n; }

  void AddArc(const Arc &arc) {
    if (arc.ilabel == 0) ++niepsilons_;
    if (arc.olabel == 0) ++noepsilons_;
    arcs_.push_back(arc);
  }

  void SetArc(const Arc &arc, size_t n) {
    if (arcs_[n].ilabel == 0) --niepsilons_;
    if (arcs_[n].olabel == 0) --noepsilons_;
    if (arc.ilabel == 0) ++niepsilons_;
    if (arc.olabel == 0) ++noepsilons_;
    arcs_[n] = arc;
  }

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

  // For state class allocation.
  void *operator new(size_t size, StateAllocator *alloc) {
    return alloc->allocate(1);
  }

  // For state destruction and memory freeing.
  static void Destroy(VectorState<A, M> *state, StateAllocator *alloc) {
    if (state) {
      state->~VectorState<A, M>();
      alloc->deallocate(state, 1);
    }
  }

 private:
  Weight final_;                       // Final weight.
  size_t niepsilons_;                  // # of input epsilons
  size_t noepsilons_;                  // # of output epsilons
  std::vector<A, ArcAllocator> arcs_;  // Arc container.
};

namespace internal {

// States are implemented by STL vectors, templated on the
// State definition. This does not manage the Fst properties.
template <class S>
class VectorFstBaseImpl : public FstImpl<typename S::Arc> {
 public:
  using State = S;
  using Arc = typename State::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  VectorFstBaseImpl() : start_(kNoStateId) {}

  ~VectorFstBaseImpl() override {
    for (StateId s = 0; s < states_.size(); ++s) {
      State::Destroy(states_[s], &state_alloc_);
    }
  }

  StateId Start() const { return start_; }

  Weight Final(StateId state) const { return states_[state]->Final(); }

  StateId NumStates() const { return states_.size(); }

  size_t NumArcs(StateId state) const { return states_[state]->NumArcs(); }

  size_t NumInputEpsilons(StateId state) const {
    return GetState(state)->NumInputEpsilons();
  }

  size_t NumOutputEpsilons(StateId state) const {
    return GetState(state)->NumOutputEpsilons();
  }

  void SetStart(StateId state) { start_ = state; }

  void SetFinal(StateId state, Weight weight) {
    states_[state]->SetFinal(std::move(weight));
  }

  StateId AddState() {
    states_.push_back(new (&state_alloc_) State(arc_alloc_));
    return states_.size() - 1;
  }

  StateId AddState(State *state) {
    states_.push_back(state);
    return states_.size() - 1;
  }

  void AddArc(StateId state, const Arc &arc) { states_[state]->AddArc(arc); }

  void DeleteStates(const std::vector<StateId> &dstates) {
    std::vector<StateId> newid(states_.size(), 0);
    for (StateId i = 0; i < dstates.size(); ++i) newid[dstates[i]] = kNoStateId;
    StateId nstates = 0;
    for (StateId state = 0; state < states_.size(); ++state) {
      if (newid[state] != kNoStateId) {
        newid[state] = nstates;
        if (state != nstates) states_[nstates] = states_[state];
        ++nstates;
      } else {
        State::Destroy(states_[state], &state_alloc_);
      }
    }
    states_.resize(nstates);
    for (StateId state = 0; state < states_.size(); ++state) {
      auto *arcs = states_[state]->MutableArcs();
      size_t narcs = 0;
      auto nieps = states_[state]->NumInputEpsilons();
      auto noeps = states_[state]->NumOutputEpsilons();
      for (size_t i = 0; i < states_[state]->NumArcs(); ++i) {
        const auto t = newid[arcs[i].nextstate];
        if (t != kNoStateId) {
          arcs[i].nextstate = t;
          if (i != narcs) arcs[narcs] = arcs[i];
          ++narcs;
        } else {
          if (arcs[i].ilabel == 0) --nieps;
          if (arcs[i].olabel == 0) --noeps;
        }
      }
      states_[state]->DeleteArcs(states_[state]->NumArcs() - narcs);
      states_[state]->SetNumInputEpsilons(nieps);
      states_[state]->SetNumOutputEpsilons(noeps);
    }
    if (Start() != kNoStateId) SetStart(newid[Start()]);
  }

  void DeleteStates() {
    for (StateId state = 0; state < states_.size(); ++state) {
      State::Destroy(states_[state], &state_alloc_);
    }
    states_.clear();
    SetStart(kNoStateId);
  }

  void DeleteArcs(StateId state, size_t n) { states_[state]->DeleteArcs(n); }

  void DeleteArcs(StateId state) { states_[state]->DeleteArcs(); }

  State *GetState(StateId state) { return states_[state]; }

  const State *GetState(StateId state) const { return states_[state]; }

  void SetState(StateId state, State *vstate) { states_[state] = vstate; }

  void ReserveStates(StateId n) { states_.reserve(n); }

  void ReserveArcs(StateId state, size_t n) { states_[state]->ReserveArcs(n); }

  // Provide information needed for generic state iterator.
  void InitStateIterator(StateIteratorData<Arc> *data) const {
    data->base = nullptr;
    data->nstates = states_.size();
  }

  // Provide information needed for generic arc iterator.
  void InitArcIterator(StateId state, ArcIteratorData<Arc> *data) const {
    data->base = nullptr;
    data->narcs = states_[state]->NumArcs();
    data->arcs = states_[state]->Arcs();
    data->ref_count = nullptr;
  }

 private:
  std::vector<State *> states_;                 // States represenation.
  StateId start_;                               // Initial state.
  typename State::StateAllocator state_alloc_;  // For state allocation.
  typename State::ArcAllocator arc_alloc_;      // For arc allocation.

  VectorFstBaseImpl(const VectorFstBaseImpl &) = delete;
  VectorFstBaseImpl &operator=(const VectorFstBaseImpl &) = delete;
};

// This is a VectorFstBaseImpl container that holds VectorStates and manages FST
// properties.
template <class S>
class VectorFstImpl : public VectorFstBaseImpl<S> {
 public:
  using State = S;
  using Arc = typename State::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;
  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::Properties;

  using VectorFstBaseImpl<S>::Start;
  using VectorFstBaseImpl<S>::NumStates;
  using VectorFstBaseImpl<S>::GetState;
  using VectorFstBaseImpl<S>::ReserveArcs;

  friend class MutableArcIterator<VectorFst<Arc, S>>;

  using BaseImpl = VectorFstBaseImpl<S>;

  VectorFstImpl() {
    SetType("vector");
    SetProperties(kNullProperties | kStaticProperties);
  }

  explicit VectorFstImpl(const Fst<Arc> &fst);

  static VectorFstImpl<S> *Read(std::istream &strm, const FstReadOptions &opts);

  void SetStart(StateId state) {
    BaseImpl::SetStart(state);
    SetProperties(SetStartProperties(Properties()));
  }

  void SetFinal(StateId state, Weight weight) {
    const auto old_weight = BaseImpl::Final(state);
    const auto properties =
        SetFinalProperties(Properties(), old_weight, weight);
    BaseImpl::SetFinal(state, std::move(weight));
    SetProperties(properties);
  }

  StateId AddState() {
    const auto state = BaseImpl::AddState();
    SetProperties(AddStateProperties(Properties()));
    return state;
  }

  void AddArc(StateId state, const Arc &arc) {
    auto *vstate = GetState(state);
    const auto *parc = vstate->NumArcs() == 0
                           ? nullptr
                           : &(vstate->GetArc(vstate->NumArcs() - 1));
    SetProperties(AddArcProperties(Properties(), state, arc, parc));
    BaseImpl::AddArc(state, arc);
  }

  void DeleteStates(const std::vector<StateId> &dstates) {
    BaseImpl::DeleteStates(dstates);
    SetProperties(DeleteStatesProperties(Properties()));
  }

  void DeleteStates() {
    BaseImpl::DeleteStates();
    SetProperties(DeleteAllStatesProperties(Properties(), kStaticProperties));
  }

  void DeleteArcs(StateId state, size_t n) {
    BaseImpl::DeleteArcs(state, n);
    SetProperties(DeleteArcsProperties(Properties()));
  }

  void DeleteArcs(StateId state) {
    BaseImpl::DeleteArcs(state);
    SetProperties(DeleteArcsProperties(Properties()));
  }

  // Properties always true of this FST class
  static constexpr uint64_t kStaticProperties = kExpanded | kMutable;

 private:
  // Minimum file format version supported.
  static constexpr int kMinFileVersion = 2;
};

template <class S>
constexpr uint64_t VectorFstImpl<S>::kStaticProperties;

template <class S>
constexpr int VectorFstImpl<S>::kMinFileVersion;

template <class S>
VectorFstImpl<S>::VectorFstImpl(const Fst<Arc> &fst) {
  SetType("vector");
  SetInputSymbols(fst.InputSymbols());
  SetOutputSymbols(fst.OutputSymbols());
  BaseImpl::SetStart(fst.Start());
  if (fst.Properties(kExpanded, false)) {
    BaseImpl::ReserveStates(CountStates(fst));
  }
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    const auto state = siter.Value();
    BaseImpl::AddState();
    BaseImpl::SetFinal(state, fst.Final(state));
    ReserveArcs(state, fst.NumArcs(state));
    for (ArcIterator<Fst<Arc>> aiter(fst, state); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      BaseImpl::AddArc(state, arc);
    }
  }
  SetProperties(fst.Properties(kCopyProperties, false) | kStaticProperties);
}

template <class S>
VectorFstImpl<S> *VectorFstImpl<S>::Read(std::istream &strm,
                                         const FstReadOptions &opts) {
  std::unique_ptr<VectorFstImpl<S>> impl(new VectorFstImpl());
  FstHeader hdr;
  if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) return nullptr;
  impl->BaseImpl::SetStart(hdr.Start());
  if (hdr.NumStates() != kNoStateId) impl->ReserveStates(hdr.NumStates());
  StateId state = 0;
  for (; hdr.NumStates() == kNoStateId || state < hdr.NumStates(); ++state) {
    Weight weight;
    if (!weight.Read(strm)) break;
    impl->BaseImpl::AddState();
    auto *vstate = impl->GetState(state);
    vstate->SetFinal(weight);
    int64_t narcs;
    ReadType(strm, &narcs);
    if (!strm) {
      LOG(ERROR) << "VectorFst::Read: Read failed: " << opts.source;
      return nullptr;
    }
    impl->ReserveArcs(state, narcs);
    for (int64_t i = 0; i < narcs; ++i) {
      Arc arc;
      ReadType(strm, &arc.ilabel);
      ReadType(strm, &arc.olabel);
      arc.weight.Read(strm);
      ReadType(strm, &arc.nextstate);
      if (!strm) {
        LOG(ERROR) << "VectorFst::Read: Read failed: " << opts.source;
        return nullptr;
      }
      impl->BaseImpl::AddArc(state, arc);
    }
  }
  if (hdr.NumStates() != kNoStateId && state != hdr.NumStates()) {
    LOG(ERROR) << "VectorFst::Read: Unexpected end of file: " << opts.source;
    return nullptr;
  }
  return impl.release();
}

}  // namespace internal

// Simple concrete, mutable FST. This class attaches interface to implementation
// and handles reference counting, delegating most methods to ImplToMutableFst.
// Also supports ReserveStates and ReserveArcs methods (cf. STL vector methods).
// The second optional template argument gives the State definition.
template <class A, class S /* = VectorState<A> */>
class VectorFst : public ImplToMutableFst<internal::VectorFstImpl<S>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using State = S;
  using Impl = internal::VectorFstImpl<State>;

  friend class StateIterator<VectorFst<Arc, State>>;
  friend class ArcIterator<VectorFst<Arc, State>>;
  friend class MutableArcIterator<VectorFst<A, S>>;

  template <class F, class G>
  friend void Cast(const F &, G *);

  VectorFst() : ImplToMutableFst<Impl>(std::make_shared<Impl>()) {}

  explicit VectorFst(const Fst<Arc> &fst)
      : ImplToMutableFst<Impl>(std::make_shared<Impl>(fst)) {}

  VectorFst(const VectorFst<Arc, State> &fst, bool safe = false)
      : ImplToMutableFst<Impl>(fst) {}

  // Get a copy of this VectorFst. See Fst<>::Copy() for further doc.
  VectorFst<Arc, State> *Copy(bool safe = false) const override {
    return new VectorFst<Arc, State>(*this, safe);
  }

  VectorFst<Arc, State> &operator=(const VectorFst<Arc, State> &fst) {
    SetImpl(fst.GetSharedImpl());
    return *this;
  }

  VectorFst<Arc, State> &operator=(const Fst<Arc> &fst) override {
    if (this != &fst) SetImpl(std::make_shared<Impl>(fst));
    return *this;
  }

  // Reads a VectorFst from an input stream, returning nullptr on error.
  static VectorFst<Arc, State> *Read(std::istream &strm,
                                     const FstReadOptions &opts) {
    auto *impl = Impl::Read(strm, opts);
    return impl ? new VectorFst<Arc, State>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  // Read a VectorFst from a file, returning nullptr on error; empty filename
  // reads from standard input.
  static VectorFst<Arc, State> *Read(const string &filename) {
    auto *impl = ImplToExpandedFst<Impl, MutableFst<Arc>>::Read(filename);
    return impl ? new VectorFst<Arc, State>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return WriteFst(*this, strm, opts);
  }

  bool Write(const string &filename) const override {
    return Fst<Arc>::WriteFile(filename);
  }

  template <class FST>
  static bool WriteFst(const FST &fst, std::ostream &strm,
                       const FstWriteOptions &opts);

  void InitStateIterator(StateIteratorData<Arc> *data) const override {
    GetImpl()->InitStateIterator(data);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetImpl()->InitArcIterator(s, data);
  }

  inline void InitMutableArcIterator(StateId s,
                                     MutableArcIteratorData<Arc> *) override;

  using ImplToMutableFst<Impl, MutableFst<Arc>>::ReserveArcs;
  using ImplToMutableFst<Impl, MutableFst<Arc>>::ReserveStates;

 private:
  using ImplToMutableFst<Impl, MutableFst<Arc>>::GetImpl;
  using ImplToMutableFst<Impl, MutableFst<Arc>>::MutateCheck;
  using ImplToMutableFst<Impl, MutableFst<Arc>>::SetImpl;

  explicit VectorFst(std::shared_ptr<Impl> impl)
      : ImplToMutableFst<Impl>(impl) {}
};

// Writes FST to file in Vector format, potentially with a pass over the machine
// before writing to compute number of states.
template <class Arc, class State>
template <class FST>
bool VectorFst<Arc, State>::WriteFst(const FST &fst, std::ostream &strm,
                                     const FstWriteOptions &opts) {
  static constexpr int file_version = 2;
  bool update_header = true;
  FstHeader hdr;
  hdr.SetStart(fst.Start());
  hdr.SetNumStates(kNoStateId);
  size_t start_offset = 0;
  if (fst.Properties(kExpanded, false) || opts.stream_write ||
      (start_offset = strm.tellp()) != -1) {
    hdr.SetNumStates(CountStates(fst));
    update_header = false;
  }
  const auto properties =
      fst.Properties(kCopyProperties, false) | Impl::kStaticProperties;
  internal::FstImpl<Arc>::WriteFstHeader(fst, strm, opts, file_version,
                                         "vector", properties, &hdr);
  StateId num_states = 0;
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    fst.Final(s).Write(strm);
    const int64_t narcs = fst.NumArcs(s);
    WriteType(strm, narcs);
    for (ArcIterator<FST> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      WriteType(strm, arc.ilabel);
      WriteType(strm, arc.olabel);
      arc.weight.Write(strm);
      WriteType(strm, arc.nextstate);
    }
    ++num_states;
  }
  strm.flush();
  if (!strm) {
    LOG(ERROR) << "VectorFst::Write: Write failed: " << opts.source;
    return false;
  }
  if (update_header) {
    hdr.SetNumStates(num_states);
    return internal::FstImpl<Arc>::UpdateFstHeader(
        fst, strm, opts, file_version, "vector", properties, &hdr,
        start_offset);
  } else {
    if (num_states != hdr.NumStates()) {
      LOG(ERROR) << "Inconsistent number of states observed during write";
      return false;
    }
  }
  return true;
}

// Specialization for VectorFst; see generic version in fst.h for sample usage
// (but use the VectorFst type instead). This version should inline.
template <class Arc, class State>
class StateIterator<VectorFst<Arc, State>> {
 public:
  using StateId = typename Arc::StateId;

  explicit StateIterator(const VectorFst<Arc, State> &fst)
      : nstates_(fst.GetImpl()->NumStates()), s_(0) {}

  bool Done() const { return s_ >= nstates_; }

  StateId Value() const { return s_; }

  void Next() { ++s_; }

  void Reset() { s_ = 0; }

 private:
  const StateId nstates_;
  StateId s_;
};

// Specialization for VectorFst; see generic version in fst.h for sample usage
// (but use the VectorFst type instead). This version should inline.
template <class Arc, class State>
class ArcIterator<VectorFst<Arc, State>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const VectorFst<Arc, State> &fst, StateId s)
      : arcs_(fst.GetImpl()->GetState(s)->Arcs()),
        narcs_(fst.GetImpl()->GetState(s)->NumArcs()),
        i_(0) {}

  bool Done() const { return i_ >= narcs_; }

  const Arc &Value() const { return arcs_[i_]; }

  void Next() { ++i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  size_t Position() const { return i_; }

  constexpr uint32_t Flags() const { return kArcValueFlags; }

  void SetFlags(uint32_t, uint32_t) {}

 private:
  const Arc *arcs_;
  size_t narcs_;
  size_t i_;
};

// Specialization for VectorFst; see generic version in mutable-fst.h for sample
// usage (but use the VectorFst type instead). This version should inline.
template <class Arc, class State>
class MutableArcIterator<VectorFst<Arc, State>>
    : public MutableArcIteratorBase<Arc> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  MutableArcIterator(VectorFst<Arc, State> *fst, StateId s) : i_(0) {
    fst->MutateCheck();
    state_ = fst->GetMutableImpl()->GetState(s);
    properties_ = &fst->GetImpl()->properties_;
  }

  bool Done() const final { return i_ >= state_->NumArcs(); }

  const Arc &Value() const final { return state_->GetArc(i_); }

  void Next() final { ++i_; }

  size_t Position() const final { return i_; }

  void Reset() final { i_ = 0; }

  void Seek(size_t a) final { i_ = a; }

  void SetValue(const Arc &arc) final {
    const auto &oarc = state_->GetArc(i_);
    if (oarc.ilabel != oarc.olabel) *properties_ &= ~kNotAcceptor;
    if (oarc.ilabel == 0) {
      *properties_ &= ~kIEpsilons;
      if (oarc.olabel == 0) *properties_ &= ~kEpsilons;
    }
    if (oarc.olabel == 0) *properties_ &= ~kOEpsilons;
    if (oarc.weight != Weight::Zero() && oarc.weight != Weight::One()) {
      *properties_ &= ~kWeighted;
    }
    state_->SetArc(arc, i_);
    if (arc.ilabel != arc.olabel) {
      *properties_ |= kNotAcceptor;
      *properties_ &= ~kAcceptor;
    }
    if (arc.ilabel == 0) {
      *properties_ |= kIEpsilons;
      *properties_ &= ~kNoIEpsilons;
      if (arc.olabel == 0) {
        *properties_ |= kEpsilons;
        *properties_ &= ~kNoEpsilons;
      }
    }
    if (arc.olabel == 0) {
      *properties_ |= kOEpsilons;
      *properties_ &= ~kNoOEpsilons;
    }
    if (arc.weight != Weight::Zero() && arc.weight != Weight::One()) {
      *properties_ |= kWeighted;
      *properties_ &= ~kUnweighted;
    }
    *properties_ &= kSetArcProperties | kAcceptor | kNotAcceptor | kEpsilons |
                    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons |
                    kNoOEpsilons | kWeighted | kUnweighted;
  }

  uint32_t Flags() const final { return kArcValueFlags; }

  void SetFlags(uint32_t, uint32_t) final {}

 private:
  State *state_;
  uint64_t *properties_;
  size_t i_;
};

// Provides information needed for the generic mutable arc iterator.
template <class Arc, class State>
inline void VectorFst<Arc, State>::InitMutableArcIterator(
    StateId s, MutableArcIteratorData<Arc> *data) {
  data->base = new MutableArcIterator<VectorFst<Arc, State>>(this, s);
}

// A useful alias when using StdArc.
using StdVectorFst = VectorFst<StdArc>;

}  // namespace fst

#endif  // FST_VECTOR_FST_H_
