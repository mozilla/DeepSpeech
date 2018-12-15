// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// An FST implementation that allows non-destructive edit operations on an
// existing FST.
//
// The EditFst class enables non-destructive edit operations on a wrapped
// ExpandedFst. The implementation uses copy-on-write semantics at the node
// level: if a user has an underlying fst on which he or she wants to perform a
// relatively small number of edits (read: mutations), then this implementation
// will copy the edited node to an internal MutableFst and perform any edits in
// situ on that copied node. This class supports all the methods of MutableFst
// except for DeleteStates(const std::vector<StateId> &); thus, new nodes may
// also be
// added, and one may add transitions from existing nodes of the wrapped fst to
// new nodes.
//
// N.B.: The documentation for Fst::Copy(true) says that its behavior is
// undefined if invoked on an fst that has already been accessed.  This class
// requires that the Fst implementation it wraps provides consistent, reliable
// behavior when its Copy(true) method is invoked, where consistent means
// the graph structure, graph properties and state numbering and do not change.
// VectorFst and CompactFst, for example, are both well-behaved in this regard.

#ifndef FST_EDIT_FST_H_
#define FST_EDIT_FST_H_

#include <string>
#include <unordered_map>
#include <vector>

#include <fst/log.h>

#include <fst/cache.h>


namespace fst {
namespace internal {

// The EditFstData class is a container for all mutable data for EditFstImpl;
// also, this class provides most of the actual implementation of what EditFst
// does (that is, most of EditFstImpl's methods delegate to methods in this, the
// EditFstData class).  Instances of this class are reference-counted and can be
// shared between otherwise independent EditFstImpl instances. This scheme
// allows EditFstImpl to implement the thread-safe, copy-on-write semantics
// required by Fst::Copy(true).
//
// template parameters:
//   A the type of arc to use
//   WrappedFstT the type of fst wrapped by the EditFst instance that
//     this EditFstData instance is backing
//   MutableFstT the type of mutable fst to use internally for edited states;
//     crucially, MutableFstT::Copy(false) *must* yield an fst that is
//     thread-safe for reading (VectorFst, for example, has this property)
template <typename Arc, typename WrappedFstT = ExpandedFst<Arc>,
          typename MutableFstT = VectorFst<Arc>>
class EditFstData {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  EditFstData() : num_new_states_(0) {}

  EditFstData(const EditFstData &other)
      : edits_(other.edits_),
        external_to_internal_ids_(other.external_to_internal_ids_),
        edited_final_weights_(other.edited_final_weights_),
        num_new_states_(other.num_new_states_) {}

  ~EditFstData() {}

  static EditFstData<Arc, WrappedFstT, MutableFstT> *Read(
      std::istream &strm, const FstReadOptions &opts);

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const {
    // Serialize all private data members of this class.
    FstWriteOptions edits_opts(opts);
    edits_opts.write_header = true;  // Force writing contained header.
    edits_.Write(strm, edits_opts);
    WriteType(strm, external_to_internal_ids_);
    WriteType(strm, edited_final_weights_);
    WriteType(strm, num_new_states_);
    if (!strm) {
      LOG(ERROR) << "EditFstData::Write: Write failed: " << opts.source;
      return false;
    }
    return true;
  }

  StateId NumNewStates() const { return num_new_states_; }

  // accessor methods for the fst holding edited states
  StateId EditedStart() const { return edits_.Start(); }

  Weight Final(StateId s, const WrappedFstT *wrapped) const {
    auto final_weight_it = GetFinalWeightIterator(s);
    if (final_weight_it == NotInFinalWeightMap()) {
      auto it = GetEditedIdMapIterator(s);
      return it == NotInEditedMap() ? wrapped->Final(s)
                                    : edits_.Final(it->second);
    } else {
      return final_weight_it->second;
    }
  }

  size_t NumArcs(StateId s, const WrappedFstT *wrapped) const {
    auto it = GetEditedIdMapIterator(s);
    return it == NotInEditedMap() ? wrapped->NumArcs(s)
                                  : edits_.NumArcs(it->second);
  }

  size_t NumInputEpsilons(StateId s, const WrappedFstT *wrapped) const {
    auto it = GetEditedIdMapIterator(s);
    return it == NotInEditedMap() ? wrapped->NumInputEpsilons(s)
                                  : edits_.NumInputEpsilons(it->second);
  }

  size_t NumOutputEpsilons(StateId s, const WrappedFstT *wrapped) const {
    auto it = GetEditedIdMapIterator(s);
    return it == NotInEditedMap() ? wrapped->NumOutputEpsilons(s)
                                  : edits_.NumOutputEpsilons(it->second);
  }

  void SetEditedProperties(uint64_t props, uint64_t mask) {
    edits_.SetProperties(props, mask);
  }

  // Non-const MutableFst operations.

  // Sets the start state for this FST.
  void SetStart(StateId s) { edits_.SetStart(s); }

  // Sets the final state for this FST.
  Weight SetFinal(StateId s, Weight w, const WrappedFstT *wrapped) {
    Weight old_weight = Final(s, wrapped);
    auto it = GetEditedIdMapIterator(s);
    // If we haven't already edited state s, don't add it to edited_ (which can
    // be expensive if s has many transitions); just use the
    // edited_final_weights_ map.
    if (it == NotInEditedMap()) {
      edited_final_weights_[s] = w;
    } else {
      edits_.SetFinal(GetEditableInternalId(s, wrapped), w);
    }
    return old_weight;
  }

  // Adds a new state to this FST, initially with no arcs.
  StateId AddState(StateId curr_num_states) {
    StateId internal_state_id = edits_.AddState();
    StateId external_state_id = curr_num_states;
    external_to_internal_ids_[external_state_id] = internal_state_id;
    num_new_states_++;
    return external_state_id;
  }

  // Adds the specified arc to the specified state of this FST.
  const Arc *AddArc(StateId s, const Arc &arc, const WrappedFstT *wrapped) {
    const auto internal_id = GetEditableInternalId(s, wrapped);
    const auto num_arcs = edits_.NumArcs(internal_id);
    ArcIterator<MutableFstT> arc_it(edits_, internal_id);
    const Arc *prev_arc = nullptr;
    if (num_arcs > 0) {
      // grab the final arc associated with this state in edits_
      arc_it.Seek(num_arcs - 1);
      prev_arc = &(arc_it.Value());
    }
    edits_.AddArc(internal_id, arc);
    return prev_arc;
  }

  void DeleteStates() {
    edits_.DeleteStates();
    num_new_states_ = 0;
    external_to_internal_ids_.clear();
    edited_final_weights_.clear();
  }

  // Removes all but the first n outgoing arcs of the specified state.
  void DeleteArcs(StateId s, size_t n, const WrappedFstT *wrapped) {
    edits_.DeleteArcs(GetEditableInternalId(s, wrapped), n);
  }

  // Removes all outgoing arcs from the specified state.
  void DeleteArcs(StateId s, const WrappedFstT *wrapped) {
    edits_.DeleteArcs(GetEditableInternalId(s, wrapped));
  }

  // End methods for non-const MutableFst operations.

  // Provides information for the generic arc iterator.
  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data,
                       const WrappedFstT *wrapped) const {
    auto id_map_it = GetEditedIdMapIterator(s);
    if (id_map_it == NotInEditedMap()) {
      VLOG(3) << "EditFstData::InitArcIterator: iterating on state " << s
              << " of original fst";
      wrapped->InitArcIterator(s, data);
    } else {
      VLOG(2) << "EditFstData::InitArcIterator: iterating on edited state " << s
              << " (internal state id: " << id_map_it->second << ")";
      edits_.InitArcIterator(id_map_it->second, data);
    }
  }

  // Provides information for the generic mutable arc iterator.
  void InitMutableArcIterator(StateId s, MutableArcIteratorData<Arc> *data,
                              const WrappedFstT *wrapped) {
    data->base = new MutableArcIterator<MutableFstT>(
        &edits_, GetEditableInternalId(s, wrapped));
  }

  // Prints out the map from external to internal state id's (for debugging
  // purposes).
  void PrintMap() {
    for (auto map_it = external_to_internal_ids_.begin();
         map_it != NotInEditedMap(); ++map_it) {
      LOG(INFO) << "(external,internal)=(" << map_it->first << ","
                << map_it->second << ")";
    }
  }

 private:
  // Returns the iterator of the map from external to internal state id's
  // of edits_ for the specified external state id.
  typename std::unordered_map<StateId, StateId>::const_iterator
      GetEditedIdMapIterator(StateId s) const {
    return external_to_internal_ids_.find(s);
  }

  typename std::unordered_map<StateId, StateId>::const_iterator
      NotInEditedMap() const {
    return external_to_internal_ids_.end();
  }

  typename std::unordered_map<StateId, Weight>::const_iterator
      GetFinalWeightIterator(StateId s) const {
    return edited_final_weights_.find(s);
  }

  typename std::unordered_map<StateId, Weight>::const_iterator
      NotInFinalWeightMap() const {
    return edited_final_weights_.end();
  }

  // Returns the internal state ID of the specified external ID if the state has
  // already been made editable, or else copies the state from wrapped_ to
  // edits_ and returns the state id of the newly editable state in edits_.
  StateId GetEditableInternalId(StateId s, const WrappedFstT *wrapped) {
    auto id_map_it = GetEditedIdMapIterator(s);
    if (id_map_it == NotInEditedMap()) {
      StateId new_internal_id = edits_.AddState();
      VLOG(2) << "EditFstData::GetEditableInternalId: editing state " << s
              << " of original fst; new internal state id:" << new_internal_id;
      external_to_internal_ids_[s] = new_internal_id;
      for (ArcIterator<Fst<Arc>> arc_iterator(*wrapped, s);
           !arc_iterator.Done(); arc_iterator.Next()) {
        edits_.AddArc(new_internal_id, arc_iterator.Value());
      }
      // Copies the final weight.
      auto final_weight_it = GetFinalWeightIterator(s);
      if (final_weight_it == NotInFinalWeightMap()) {
        edits_.SetFinal(new_internal_id, wrapped->Final(s));
      } else {
        edits_.SetFinal(new_internal_id, final_weight_it->second);
        edited_final_weights_.erase(s);
      }
      return new_internal_id;
    } else {
      return id_map_it->second;
    }
  }

  // A mutable FST (by default, a VectorFst) to contain new states, and/or
  // copies of states from a wrapped ExpandedFst that have been modified in
  // some way.
  MutableFstT edits_;
  // A mapping from external state IDs to the internal IDs of states that
  // appear in edits_.
  std::unordered_map<StateId, StateId> external_to_internal_ids_;
  // A mapping from external state IDs to final state weights assigned to
  // those states.  The states in this map are *only* those whose final weight
  // has been modified; if any other part of the state has been modified,
  // the entire state is copied to edits_, and all modifications reside there.
  std::unordered_map<StateId, Weight> edited_final_weights_;
  // The number of new states added to this mutable fst impl, which is <= the
  // number of states in edits_ (since edits_ contains both edited *and* new
  // states).
  StateId num_new_states_;
};

// EditFstData method implementations: just the Read method.
template <typename A, typename WrappedFstT, typename MutableFstT>
EditFstData<A, WrappedFstT, MutableFstT> *
EditFstData<A, WrappedFstT, MutableFstT>::Read(std::istream &strm,
                                               const FstReadOptions &opts) {
  auto *data = new EditFstData<A, WrappedFstT, MutableFstT>();
  // next read in MutabelFstT machine that stores edits
  FstReadOptions edits_opts(opts);
  // Contained header was written out, so read it in.
  edits_opts.header = nullptr;

  // Because our internal representation of edited states is a solid object
  // of type MutableFstT (defaults to VectorFst<A>) and not a pointer,
  // and because the static Read method allocates a new object on the heap,
  // we need to call Read, check if there was a failure, use
  // MutableFstT::operator= to assign the object (not the pointer) to the
  // edits_ data member (which will increase the ref count by 1 on the impl)
  // and, finally, delete the heap-allocated object.
  std::unique_ptr<MutableFstT> edits(MutableFstT::Read(strm, edits_opts));
  if (!edits) return nullptr;
  data->edits_ = *edits;
  edits.reset();
  // Finally, reads in rest of private data members.
  ReadType(strm, &data->external_to_internal_ids_);
  ReadType(strm, &data->edited_final_weights_);
  ReadType(strm, &data->num_new_states_);
  if (!strm) {
    LOG(ERROR) << "EditFst::Read: read failed: " << opts.source;
    return nullptr;
  }
  return data;
}

// This class enables non-destructive edit operations on a wrapped ExpandedFst.
// The implementation uses copy-on-write semantics at the node level: if a user
// has an underlying fst on which he or she wants to perform a relatively small
// number of edits (read: mutations), then this implementation will copy the
// edited node to an internal MutableFst and perform any edits in situ on that
// copied node. This class supports all the methods of MutableFst except for
// DeleteStates(const std::vector<StateId> &); thus, new nodes may also be
// added, and
// one may add transitions from existing nodes of the wrapped fst to new nodes.
//
// template parameters:
//   A the type of arc to use
//   WrappedFstT the type of fst wrapped by the EditFst instance that
//     this EditFstImpl instance is backing
//   MutableFstT the type of mutable fst to use internally for edited states;
//     crucially, MutableFstT::Copy(false) *must* yield an fst that is
//     thread-safe for reading (VectorFst, for example, has this property)
template <typename A, typename WrappedFstT = ExpandedFst<A>,
          typename MutableFstT = VectorFst<A>>
class EditFstImpl : public FstImpl<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;
  using FstImpl<Arc>::WriteHeader;

  // Constructs an editable FST implementation with no states. Effectively, this
  // initially-empty fst will in every way mimic the behavior of a
  // VectorFst---more precisely, a VectorFstImpl instance---but with slightly
  // slower performance (by a constant factor), due to the fact that
  // this class maintains a mapping between external state id's and
  // their internal equivalents.
  EditFstImpl() : wrapped_(new MutableFstT()) {
    FstImpl<Arc>::SetType("edit");
    InheritPropertiesFromWrapped();
    data_ = std::make_shared<EditFstData<Arc, WrappedFstT, MutableFstT>>();
  }

  // Wraps the specified ExpandedFst. This constructor requires that the
  // specified Fst is an ExpandedFst instance. This requirement is only enforced
  // at runtime. (See below for the reason.)
  //
  // This library uses the pointer-to-implementation or "PIMPL" design pattern.
  // In particular, to make it convenient to bind an implementation class to its
  // interface, there are a pair of template "binder" classes, one for immutable
  // and one for mutable fst's (ImplToFst and ImplToMutableFst, respectively).
  // As it happens, the API for the ImplToMutableFst<I,F> class requires that
  // the implementation class--the template parameter "I"--have a constructor
  // taking a const Fst<A> reference.  Accordingly, the constructor here must
  // perform a static_cast to the WrappedFstT type required by EditFst and
  // therefore EditFstImpl.
  explicit EditFstImpl(const Fst<Arc> &wrapped)
      : wrapped_(static_cast<WrappedFstT *>(wrapped.Copy())) {
    FstImpl<Arc>::SetType("edit");
    data_ = std::make_shared<EditFstData<Arc, WrappedFstT, MutableFstT>>();
    // have edits_ inherit all properties from wrapped_
    data_->SetEditedProperties(wrapped_->Properties(kFstProperties, false),
                               kFstProperties);
    InheritPropertiesFromWrapped();
  }

  // A copy constructor for this implementation class, used to implement
  // the Copy() method of the Fst interface.
  EditFstImpl(const EditFstImpl &impl)
      : FstImpl<Arc>(),
        wrapped_(static_cast<WrappedFstT *>(impl.wrapped_->Copy(true))),
        data_(impl.data_) {
    SetProperties(impl.Properties());
  }

  // const Fst/ExpandedFst operations, declared in the Fst and ExpandedFst
  // interfaces
  StateId Start() const {
    const auto edited_start = data_->EditedStart();
    return edited_start == kNoStateId ? wrapped_->Start() : edited_start;
  }

  Weight Final(StateId s) const { return data_->Final(s, wrapped_.get()); }

  size_t NumArcs(StateId s) const { return data_->NumArcs(s, wrapped_.get()); }

  size_t NumInputEpsilons(StateId s) const {
    return data_->NumInputEpsilons(s, wrapped_.get());
  }

  size_t NumOutputEpsilons(StateId s) const {
    return data_->NumOutputEpsilons(s, wrapped_.get());
  }

  StateId NumStates() const {
    return wrapped_->NumStates() + data_->NumNewStates();
  }

  static EditFstImpl<Arc, WrappedFstT, MutableFstT> *Read(
      std::istream &strm, const FstReadOptions &opts);

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const {
    FstHeader hdr;
    hdr.SetStart(Start());
    hdr.SetNumStates(NumStates());
    FstWriteOptions header_opts(opts);
    // Allows the contained FST to hold any symbols.
    header_opts.write_isymbols = false;
    header_opts.write_osymbols = false;
    WriteHeader(strm, header_opts, kFileVersion, &hdr);
    // First, serializes the wrapped FST to stream.
    FstWriteOptions wrapped_opts(opts);
    // Forcse writing the contained header.
    wrapped_opts.write_header = true;
    wrapped_->Write(strm, wrapped_opts);
    data_->Write(strm, opts);
    strm.flush();
    if (!strm) {
      LOG(ERROR) << "EditFst::Write: Write failed: " << opts.source;
      return false;
    }
    return true;
  }

  // Sets the start state for this FST.
  void SetStart(StateId s) {
    MutateCheck();
    data_->SetStart(s);
    SetProperties(SetStartProperties(FstImpl<Arc>::Properties()));
  }

  // Sets the final state for this fst.
  void SetFinal(StateId s, Weight weight) {
    MutateCheck();
    Weight old_weight = data_->SetFinal(s, weight, wrapped_.get());
    SetProperties(
        SetFinalProperties(FstImpl<Arc>::Properties(), old_weight, weight));
  }

  // Adds a new state to this fst, initially with no arcs.
  StateId AddState() {
    MutateCheck();
    SetProperties(AddStateProperties(FstImpl<Arc>::Properties()));
    return data_->AddState(NumStates());
  }

  // Adds the specified arc to the specified state of this fst.
  void AddArc(StateId s, const Arc &arc) {
    MutateCheck();
    const auto *prev_arc = data_->AddArc(s, arc, wrapped_.get());
    SetProperties(
        AddArcProperties(FstImpl<Arc>::Properties(), s, arc, prev_arc));
  }

  void DeleteStates(const std::vector<StateId> &dstates) {
    FSTERROR() << ": EditFstImpl::DeleteStates(const std::vector<StateId>&): "
               << " not implemented";
    SetProperties(kError, kError);
  }

  // Deletes all states in this fst.
  void DeleteStates();

  // Removes all but the first n outgoing arcs of the specified state.
  void DeleteArcs(StateId s, size_t n) {
    MutateCheck();
    data_->DeleteArcs(s, n, wrapped_.get());
    SetProperties(DeleteArcsProperties(FstImpl<Arc>::Properties()));
  }

  // Removes all outgoing arcs from the specified state.
  void DeleteArcs(StateId s) {
    MutateCheck();
    data_->DeleteArcs(s, wrapped_.get());
    SetProperties(DeleteArcsProperties(FstImpl<Arc>::Properties()));
  }

  void ReserveStates(StateId s) {}

  void ReserveArcs(StateId s, size_t n) {}

  // Ends non-const MutableFst operations.

  // Provides information for the generic state iterator.
  void InitStateIterator(StateIteratorData<Arc> *data) const {
    data->base = nullptr;
    data->nstates = NumStates();
  }

  // Provides information for the generic arc iterator.
  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const {
    data_->InitArcIterator(s, data, wrapped_.get());
  }

  // Provides information for the generic mutable arc iterator.
  void InitMutableArcIterator(StateId s, MutableArcIteratorData<Arc> *data) {
    MutateCheck();
    data_->InitMutableArcIterator(s, data, wrapped_.get());
  }

 private:
  // Properties always true of this FST class.
  static constexpr uint64_t kStaticProperties = kExpanded | kMutable;
  // Current file format version.
  static constexpr int kFileVersion = 2;
  // Minimum file format version supported
  static constexpr int kMinFileVersion = 2;

  // Causes this FST to inherit all the properties from its wrapped FST, except
  // for the two properties that always apply to EditFst instances: kExpanded
  // and kMutable.
  void InheritPropertiesFromWrapped() {
    SetProperties(wrapped_->Properties(kCopyProperties, false) |
                  kStaticProperties);
    SetInputSymbols(wrapped_->InputSymbols());
    SetOutputSymbols(wrapped_->OutputSymbols());
  }

  // This method ensures that any operations that alter the mutable data
  // portion of this EditFstImpl cause the data_ member to be copied when its
  // reference count is greater than 1.  Note that this method is distinct from
  // MutableFst::Mutate, which gets invoked whenever one of the basic mutation
  // methods defined in MutableFst is invoked, such as SetInputSymbols.
  // The MutateCheck here in EditFstImpl is invoked whenever one of the
  // mutating methods specifically related to the types of edits provided
  // by EditFst is performed, such as changing an arc of an existing state
  // of the wrapped fst via a MutableArcIterator, or adding a new state via
  // AddState().
  void MutateCheck() {
    if (!data_.unique()) {
      data_ =
          std::make_shared<EditFstData<Arc, WrappedFstT, MutableFstT>>(*data_);
    }
  }

  // The FST that this FST wraps. The purpose of this class is to enable
  // non-destructive edits on this wrapped FST.
  std::unique_ptr<const WrappedFstT> wrapped_;
  // The mutable data for this EditFst instance, with delegates for all the
  // methods that can mutate data.
  std::shared_ptr<EditFstData<Arc, WrappedFstT, MutableFstT>> data_;
};

template <typename Arc, typename WrappedFstT, typename MutableFstT>
constexpr uint64_t EditFstImpl<Arc, WrappedFstT, MutableFstT>::kStaticProperties;

template <typename Arc, typename WrappedFstT, typename MutableFstT>
constexpr int EditFstImpl<Arc, WrappedFstT, MutableFstT>::kFileVersion;

template <typename Arc, typename WrappedFstT, typename MutableFstT>
constexpr int EditFstImpl<Arc, WrappedFstT, MutableFstT>::kMinFileVersion;

template <typename Arc, typename WrappedFstT, typename MutableFstT>
inline void EditFstImpl<Arc, WrappedFstT, MutableFstT>::DeleteStates() {
  data_->DeleteStates();
  // we are deleting all states, so just forget about pointer to wrapped_
  // and do what default constructor does: set wrapped_ to a new VectorFst
  wrapped_.reset(new MutableFstT());
  const auto new_props =
      DeleteAllStatesProperties(FstImpl<Arc>::Properties(), kStaticProperties);
  FstImpl<Arc>::SetProperties(new_props);
}

template <typename Arc, typename WrappedFstT, typename MutableFstT>
EditFstImpl<Arc, WrappedFstT, MutableFstT> *
EditFstImpl<Arc, WrappedFstT, MutableFstT>::Read(std::istream &strm,
                                                 const FstReadOptions &opts) {
  auto *impl = new EditFstImpl();
  FstHeader hdr;
  if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) return nullptr;
  impl->SetStart(hdr.Start());
  // Reads in wrapped FST.
  FstReadOptions wrapped_opts(opts);
  // Contained header was written out, so reads it in too.
  wrapped_opts.header = nullptr;
  std::unique_ptr<Fst<Arc>> wrapped_fst(Fst<Arc>::Read(strm, wrapped_opts));
  if (!wrapped_fst) return nullptr;
  impl->wrapped_.reset(static_cast<WrappedFstT *>(wrapped_fst.release()));
  impl->data_ = std::shared_ptr<EditFstData<Arc, WrappedFstT, MutableFstT>>(
      EditFstData<Arc, WrappedFstT, MutableFstT>::Read(strm, opts));
  if (!impl->data_) return nullptr;
  return impl;
}

}  // namespace internal

// Concrete, editable FST.  This class attaches interface to implementation.
template <typename A, typename WrappedFstT = ExpandedFst<A>,
          typename MutableFstT = VectorFst<A>>
class EditFst : public ImplToMutableFst<
                    internal::EditFstImpl<A, WrappedFstT, MutableFstT>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using Impl = internal::EditFstImpl<Arc, WrappedFstT, MutableFstT>;

  friend class MutableArcIterator<EditFst<Arc, WrappedFstT, MutableFstT>>;

  EditFst() : ImplToMutableFst<Impl>(std::make_shared<Impl>()) {}

  explicit EditFst(const Fst<Arc> &fst)
      : ImplToMutableFst<Impl>(std::make_shared<Impl>(fst)) {}

  explicit EditFst(const WrappedFstT &fst)
      : ImplToMutableFst<Impl>(std::make_shared<Impl>(fst)) {}

  // See Fst<>::Copy() for doc.
  EditFst(const EditFst<Arc, WrappedFstT, MutableFstT> &fst, bool safe = false)
      : ImplToMutableFst<Impl>(fst, safe) {}

  ~EditFst() override {}

  // Gets a copy of this EditFst. See Fst<>::Copy() for further doc.
  EditFst<Arc, WrappedFstT, MutableFstT> *Copy(
      bool safe = false) const override {
    return new EditFst<Arc, WrappedFstT, MutableFstT>(*this, safe);
  }

  EditFst<Arc, WrappedFstT, MutableFstT> &operator=(
      const EditFst<Arc, WrappedFstT, MutableFstT> &fst) {
    SetImpl(fst.GetSharedImpl());
    return *this;
  }

  EditFst<Arc, WrappedFstT, MutableFstT> &operator=(
      const Fst<Arc> &fst) override {
    SetImpl(std::make_shared<Impl>(fst));
    return *this;
  }

  // Reads an EditFst from an input stream, returning nullptr on error.
  static EditFst<Arc, WrappedFstT, MutableFstT> *Read(
      std::istream &strm, const FstReadOptions &opts) {
    auto *impl = Impl::Read(strm, opts);
    return impl ? new EditFst<Arc>(std::shared_ptr<Impl>(impl)) : nullptr;
  }

  // Reads an EditFst from a file, returning nullptr on error. If the filename
  // argument is an empty string, it reads from standard input.
  static EditFst<Arc, WrappedFstT, MutableFstT> *Read(const string &filename) {
    auto *impl = ImplToExpandedFst<Impl, MutableFst<Arc>>::Read(filename);
    return impl ? new EditFst<Arc, WrappedFstT, MutableFstT>(
                      std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return GetImpl()->Write(strm, opts);
  }

  bool Write(const string &filename) const override {
    return Fst<Arc>::WriteFile(filename);
  }

  void InitStateIterator(StateIteratorData<Arc> *data) const override {
    GetImpl()->InitStateIterator(data);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetImpl()->InitArcIterator(s, data);
  }

  void InitMutableArcIterator(StateId s,
                              MutableArcIteratorData<A> *data) override {
    GetMutableImpl()->InitMutableArcIterator(s, data);
  }

 private:
  explicit EditFst(std::shared_ptr<Impl> impl) : ImplToMutableFst<Impl>(impl) {}

  using ImplToFst<Impl, MutableFst<Arc>>::GetImpl;
  using ImplToFst<Impl, MutableFst<Arc>>::GetMutableImpl;
  using ImplToFst<Impl, MutableFst<Arc>>::SetImpl;
};

}  // namespace fst

#endif  // FST_EDIT_FST_H_
