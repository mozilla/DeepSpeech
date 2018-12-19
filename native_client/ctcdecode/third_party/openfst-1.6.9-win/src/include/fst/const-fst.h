// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Simple concrete immutable FST whose states and arcs are each stored in
// single arrays.

#ifndef FST_CONST_FST_H_
#define FST_CONST_FST_H_

#include <climits>
#include <string>
#include <vector>

// Google-only...
// ...Google-only
#include <fst/log.h>

#include <fst/expanded-fst.h>
#include <fst/fst-decl.h>
#include <fst/mapped-file.h>
#include <fst/test-properties.h>
#include <fst/util.h>


namespace fst {

template <class A, class Unsigned>
class ConstFst;

template <class F, class G>
void Cast(const F &, G *);

namespace internal {

// States and arcs each implemented by single arrays, templated on the
// Arc definition. Unsigned is used to represent indices into the arc array.
template <class A, class Unsigned>
class ConstFstImpl : public FstImpl<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;
  using FstImpl<A>::SetType;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::Properties;

  ConstFstImpl()
      : states_(nullptr),
        arcs_(nullptr),
        nstates_(0),
        narcs_(0),
        start_(kNoStateId) {
    string type = "const";
    if (sizeof(Unsigned) != sizeof(uint32_t)) {
      type += std::to_string(CHAR_BIT * sizeof(Unsigned));
    }
    SetType(type);
    SetProperties(kNullProperties | kStaticProperties);
  }

  explicit ConstFstImpl(const Fst<Arc> &fst);

  StateId Start() const { return start_; }

  Weight Final(StateId s) const { return states_[s].weight; }

  StateId NumStates() const { return nstates_; }

  size_t NumArcs(StateId s) const { return states_[s].narcs; }

  size_t NumInputEpsilons(StateId s) const { return states_[s].niepsilons; }

  size_t NumOutputEpsilons(StateId s) const { return states_[s].noepsilons; }

  static ConstFstImpl<Arc, Unsigned> *Read(std::istream &strm,
                                           const FstReadOptions &opts);

  const Arc *Arcs(StateId s) const { return arcs_ + states_[s].pos; }

  // Provide information needed for generic state iterator.
  void InitStateIterator(StateIteratorData<Arc> *data) const {
    data->base = nullptr;
    data->nstates = nstates_;
  }

  // Provide information needed for the generic arc iterator.
  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const {
    data->base = nullptr;
    data->arcs = arcs_ + states_[s].pos;
    data->narcs = states_[s].narcs;
    data->ref_count = nullptr;
  }

 private:
  // Used to find narcs_ and nstates_ in Write.
  friend class ConstFst<Arc, Unsigned>;

  // States implemented by array *states_ below, arcs by (single) *arcs_.
  struct ConstState {
    Weight weight;        // Final weight.
    Unsigned pos;         // Start of state's arcs in *arcs_.
    Unsigned narcs;       // Number of arcs (per state).
    Unsigned niepsilons;  // Number of input epsilons.
    Unsigned noepsilons;  // Number of output epsilons.

    ConstState() : weight(Weight::Zero()) {}
  };

  // Properties always true of this FST class.
  static constexpr uint64_t kStaticProperties = kExpanded;
  // Current unaligned file format version. The unaligned version was added and
  // made the default since the aligned version does not work on pipes.
  static constexpr int kFileVersion = 2;
  // Current aligned file format version.
  static constexpr int kAlignedFileVersion = 1;
  // Minimum file format version supported.
  static constexpr int kMinFileVersion = 1;

  std::unique_ptr<MappedFile> states_region_;  // Mapped file for states.
  std::unique_ptr<MappedFile> arcs_region_;    // Mapped file for arcs.
  ConstState *states_;                         // States representation.
  Arc *arcs_;                                  // Arcs representation.
  StateId nstates_;                            // Number of states.
  size_t narcs_;                               // Number of arcs.
  StateId start_;                              // Initial state.

  ConstFstImpl(const ConstFstImpl &) = delete;
  ConstFstImpl &operator=(const ConstFstImpl &) = delete;
};

template <class Arc, class Unsigned>
constexpr uint64_t ConstFstImpl<Arc, Unsigned>::kStaticProperties;

template <class Arc, class Unsigned>
constexpr int ConstFstImpl<Arc, Unsigned>::kFileVersion;

template <class Arc, class Unsigned>
constexpr int ConstFstImpl<Arc, Unsigned>::kAlignedFileVersion;

template <class Arc, class Unsigned>
constexpr int ConstFstImpl<Arc, Unsigned>::kMinFileVersion;

template <class Arc, class Unsigned>
ConstFstImpl<Arc, Unsigned>::ConstFstImpl(const Fst<Arc> &fst)
    : nstates_(0), narcs_(0) {
  string type = "const";
  if (sizeof(Unsigned) != sizeof(uint32_t)) {
    type += std::to_string(CHAR_BIT * sizeof(Unsigned));
  }
  SetType(type);
  SetInputSymbols(fst.InputSymbols());
  SetOutputSymbols(fst.OutputSymbols());
  start_ = fst.Start();
  // Counts states and arcs.
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    ++nstates_;
    for (ArcIterator<Fst<Arc>> aiter(fst, siter.Value()); !aiter.Done();
         aiter.Next()) {
      ++narcs_;
    }
  }
  states_region_.reset(MappedFile::Allocate(nstates_ * sizeof(*states_)));
  arcs_region_.reset(MappedFile::Allocate(narcs_ * sizeof(*arcs_)));
  states_ = reinterpret_cast<ConstState *>(states_region_->mutable_data());
  arcs_ = reinterpret_cast<Arc *>(arcs_region_->mutable_data());
  size_t pos = 0;
  for (StateId s = 0; s < nstates_; ++s) {
    states_[s].weight = fst.Final(s);
    states_[s].pos = pos;
    states_[s].narcs = 0;
    states_[s].niepsilons = 0;
    states_[s].noepsilons = 0;
    for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      ++states_[s].narcs;
      if (arc.ilabel == 0) ++states_[s].niepsilons;
      if (arc.olabel == 0) ++states_[s].noepsilons;
      arcs_[pos] = arc;
      ++pos;
    }
  }
  const auto props =
      fst.Properties(kMutable, false)
          ? fst.Properties(kCopyProperties, true)
          : CheckProperties(
                fst, kCopyProperties & ~kWeightedCycles & ~kUnweightedCycles,
                kCopyProperties);
  SetProperties(props | kStaticProperties);
}

template <class Arc, class Unsigned>
ConstFstImpl<Arc, Unsigned> *ConstFstImpl<Arc, Unsigned>::Read(
    std::istream &strm, const FstReadOptions &opts) {
  using ConstState = typename ConstFstImpl<Arc, Unsigned>::ConstState;
  std::unique_ptr<ConstFstImpl<Arc, Unsigned>> impl(
      new ConstFstImpl<Arc, Unsigned>());
  FstHeader hdr;
  if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) return nullptr;
  impl->start_ = hdr.Start();
  impl->nstates_ = hdr.NumStates();
  impl->narcs_ = hdr.NumArcs();
  // Ensures compatibility.
  if (hdr.Version() == kAlignedFileVersion) {
    hdr.SetFlags(hdr.GetFlags() | FstHeader::IS_ALIGNED);
  }
  if ((hdr.GetFlags() & FstHeader::IS_ALIGNED) && !AlignInput(strm)) {
    LOG(ERROR) << "ConstFst::Read: Alignment failed: " << opts.source;
    return nullptr;
  }
  size_t b = impl->nstates_ * sizeof(ConstState);
  impl->states_region_.reset(
      MappedFile::Map(&strm, opts.mode == FstReadOptions::MAP, opts.source, b));
  if (!strm || !impl->states_region_) {
    LOG(ERROR) << "ConstFst::Read: Read failed: " << opts.source;
    return nullptr;
  }
  impl->states_ =
      reinterpret_cast<ConstState *>(impl->states_region_->mutable_data());
  if ((hdr.GetFlags() & FstHeader::IS_ALIGNED) && !AlignInput(strm)) {
    LOG(ERROR) << "ConstFst::Read: Alignment failed: " << opts.source;
    return nullptr;
  }
  b = impl->narcs_ * sizeof(Arc);
  impl->arcs_region_.reset(
      MappedFile::Map(&strm, opts.mode == FstReadOptions::MAP, opts.source, b));
  if (!strm || !impl->arcs_region_) {
    LOG(ERROR) << "ConstFst::Read: Read failed: " << opts.source;
    return nullptr;
  }
  impl->arcs_ = reinterpret_cast<Arc *>(impl->arcs_region_->mutable_data());
  return impl.release();
}

}  // namespace internal

// Simple concrete immutable FST. This class attaches interface to
// implementation and handles reference counting, delegating most methods to
// ImplToExpandedFst. The unsigned type U is used to represent indices into the
// arc array (default declared in fst-decl.h).
template <class A, class Unsigned>
class ConstFst : public ImplToExpandedFst<internal::ConstFstImpl<A, Unsigned>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using Impl = internal::ConstFstImpl<A, Unsigned>;
  using ConstState = typename Impl::ConstState;

  friend class StateIterator<ConstFst<Arc, Unsigned>>;
  friend class ArcIterator<ConstFst<Arc, Unsigned>>;

  template <class F, class G>
  void friend Cast(const F &, G *);

  ConstFst() : ImplToExpandedFst<Impl>(std::make_shared<Impl>()) {}

  explicit ConstFst(const Fst<Arc> &fst)
      : ImplToExpandedFst<Impl>(std::make_shared<Impl>(fst)) {}

  ConstFst(const ConstFst<A, Unsigned> &fst, bool safe = false)
      : ImplToExpandedFst<Impl>(fst) {}

  // Gets a copy of this ConstFst. See Fst<>::Copy() for further doc.
  ConstFst<A, Unsigned> *Copy(bool safe = false) const override {
    return new ConstFst<A, Unsigned>(*this, safe);
  }

  // Reads a ConstFst from an input stream, returning nullptr on error.
  static ConstFst<A, Unsigned> *Read(std::istream &strm,
                                     const FstReadOptions &opts) {
    auto *impl = Impl::Read(strm, opts);
    return impl ? new ConstFst<A, Unsigned>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  // Read a ConstFst from a file; return nullptr on error; empty filename reads
  // from standard input.
  static ConstFst<A, Unsigned> *Read(const string &filename) {
    auto *impl = ImplToExpandedFst<Impl>::Read(filename);
    return impl ? new ConstFst<A, Unsigned>(std::shared_ptr<Impl>(impl))
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

 private:
  explicit ConstFst(std::shared_ptr<Impl> impl)
      : ImplToExpandedFst<Impl>(impl) {}

  using ImplToFst<Impl, ExpandedFst<Arc>>::GetImpl;

  // Uses overloading to extract the type of the argument.
  static const Impl *GetImplIfConstFst(const ConstFst &const_fst) {
    return const_fst.GetImpl();
  }

  // NB: this does not give privileged treatment to subtypes of ConstFst.
  template <typename FST>
  static Impl *GetImplIfConstFst(const FST &fst) {
    return nullptr;
  }

  ConstFst &operator=(const ConstFst &) = delete;
};

// Writes FST in Const format, potentially with a pass over the machine before
// writing to compute number of states and arcs.
template <class Arc, class Unsigned>
template <class FST>
bool ConstFst<Arc, Unsigned>::WriteFst(const FST &fst, std::ostream &strm,
                                       const FstWriteOptions &opts) {
  const auto file_version =
      opts.align ? internal::ConstFstImpl<Arc, Unsigned>::kAlignedFileVersion
                 : internal::ConstFstImpl<Arc, Unsigned>::kFileVersion;
  size_t num_arcs = 0;    // To silence -Wsometimes-uninitialized warnings.
  size_t num_states = 0;  // Ditto.
  size_t start_offset = 0;
  bool update_header = true;
  if (const auto *impl = GetImplIfConstFst(fst)) {
    num_arcs = impl->narcs_;
    num_states = impl->nstates_;
    update_header = false;
  } else if (opts.stream_write || (start_offset = strm.tellp()) == -1) {
    // precompute values needed for header when we cannot seek to rewrite it.
    num_arcs = 0;
    num_states = 0;
    for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
      num_arcs += fst.NumArcs(siter.Value());
      ++num_states;
    }
    update_header = false;
  }
  FstHeader hdr;
  hdr.SetStart(fst.Start());
  hdr.SetNumStates(num_states);
  hdr.SetNumArcs(num_arcs);
  string type = "const";
  if (sizeof(Unsigned) != sizeof(uint32_t)) {
    type += std::to_string(CHAR_BIT * sizeof(Unsigned));
  }
  const auto properties =
      fst.Properties(kCopyProperties, true) |
      internal::ConstFstImpl<Arc, Unsigned>::kStaticProperties;
  internal::FstImpl<Arc>::WriteFstHeader(fst, strm, opts, file_version, type,
                                         properties, &hdr);
  if (opts.align && !AlignOutput(strm)) {
    LOG(ERROR) << "Could not align file during write after header";
    return false;
  }
  size_t pos = 0;
  size_t states = 0;
  typename ConstFst<Arc, Unsigned>::ConstState state;
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    state.weight = fst.Final(s);
    state.pos = pos;
    state.narcs = fst.NumArcs(s);
    state.niepsilons = fst.NumInputEpsilons(s);
    state.noepsilons = fst.NumOutputEpsilons(s);
    strm.write(reinterpret_cast<const char *>(&state), sizeof(state));
    pos += state.narcs;
    ++states;
  }
  hdr.SetNumStates(states);
  hdr.SetNumArcs(pos);
  if (opts.align && !AlignOutput(strm)) {
    LOG(ERROR) << "Could not align file during write after writing states";
  }
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    for (ArcIterator<FST> aiter(fst, siter.Value()); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
// Google-only...
#ifdef MEMORY_SANITIZER
      // arc may contain padding which has unspecified contents. Tell MSAN to
      // not complain about it when writing it to a file.
      ANNOTATE_MEMORY_IS_INITIALIZED(reinterpret_cast<const char *>(&arc),
                                     sizeof(arc));
#endif
      // ...Google-only
      strm.write(reinterpret_cast<const char *>(&arc), sizeof(arc));
    }
  }
  strm.flush();
  if (!strm) {
    LOG(ERROR) << "ConstFst::WriteFst: write failed: " << opts.source;
    return false;
  }
  if (update_header) {
    return internal::FstImpl<Arc>::UpdateFstHeader(
        fst, strm, opts, file_version, type, properties, &hdr, start_offset);
  } else {
    if (hdr.NumStates() != num_states) {
      LOG(ERROR) << "Inconsistent number of states observed during write";
      return false;
    }
    if (hdr.NumArcs() != num_arcs) {
      LOG(ERROR) << "Inconsistent number of arcs observed during write";
      return false;
    }
  }
  return true;
}

// Specialization for ConstFst; see generic version in fst.h for sample usage
// (but use the ConstFst type instead). This version should inline.
template <class Arc, class Unsigned>
class StateIterator<ConstFst<Arc, Unsigned>> {
 public:
  using StateId = typename Arc::StateId;

  explicit StateIterator(const ConstFst<Arc, Unsigned> &fst)
      : nstates_(fst.GetImpl()->NumStates()), s_(0) {}

  bool Done() const { return s_ >= nstates_; }

  StateId Value() const { return s_; }

  void Next() { ++s_; }

  void Reset() { s_ = 0; }

 private:
  const StateId nstates_;
  StateId s_;
};

// Specialization for ConstFst; see generic version in fst.h for sample usage
// (but use the ConstFst type instead). This version should inline.
template <class Arc, class Unsigned>
class ArcIterator<ConstFst<Arc, Unsigned>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const ConstFst<Arc, Unsigned> &fst, StateId s)
      : arcs_(fst.GetImpl()->Arcs(s)),
        narcs_(fst.GetImpl()->NumArcs(s)),
        i_(0) {}

  bool Done() const { return i_ >= narcs_; }

  const Arc &Value() const { return arcs_[i_]; }

  void Next() { ++i_; }

  size_t Position() const { return i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  constexpr uint32_t Flags() const { return kArcValueFlags; }

  void SetFlags(uint32_t, uint32_t) {}

 private:
  const Arc *arcs_;
  size_t narcs_;
  size_t i_;
};

// A useful alias when using StdArc.
using StdConstFst = ConstFst<StdArc>;

}  // namespace fst

#endif  // FST_CONST_FST_H_
