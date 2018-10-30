// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to add a matcher to an FST.

#ifndef FST_MATCHER_FST_H_
#define FST_MATCHER_FST_H_

#include <memory>
#include <string>

#include <fst/add-on.h>
#include <fst/const-fst.h>
#include <fst/lookahead-matcher.h>


namespace fst {

// Writeable matchers have the same interface as Matchers (as defined in
// matcher.h) along with the following additional methods:
//
// template <class F>
// class Matcher {
//  public:
//   using FST = F;
//   ...
//   using MatcherData = ...;   // Initialization data.
//
//   // Constructor with additional argument for external initialization data;
//   // matcher increments its reference count on construction and decrements
//   // the reference count, and deletes once the reference count has reached
//   // zero.
//   Matcher(const FST &fst, MatchType type, MatcherData *data);
//
//   // Returns pointer to initialization data that can be passed to a Matcher
//   // constructor.
//   MatcherData *GetData() const;
// };

// The matcher initialization data class must also provide the following
// interface:
//
// class MatcherData {
// public:
//   // Required copy constructor.
//   MatcherData(const MatcherData &);
//
//   // Required I/O methods.
//   static MatcherData *Read(std::istream &istrm, const FstReadOptions &opts);
//   bool Write(std::ostream &ostrm, const FstWriteOptions &opts) const;
// };

// Trivial (no-op) MatcherFst initializer functor.
template <class M>
class NullMatcherFstInit {
 public:
  using MatcherData = typename M::MatcherData;
  using Data = AddOnPair<MatcherData, MatcherData>;
  using Impl = internal::AddOnImpl<typename M::FST, Data>;

  explicit NullMatcherFstInit(std::shared_ptr<Impl> *) {}
};

// Class adding a matcher to an FST type. Creates a new FST whose name is given
// by N. An optional functor Init can be used to initialize the FST. The Data
// template parameter allows the user to select the type of the add-on.
template <
    class F, class M, const char *Name, class Init = NullMatcherFstInit<M>,
    class Data = AddOnPair<typename M::MatcherData, typename M::MatcherData>>
class MatcherFst : public ImplToExpandedFst<internal::AddOnImpl<F, Data>> {
 public:
  using FST = F;
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;

  using FstMatcher = M;
  using MatcherData = typename FstMatcher::MatcherData;

  using Impl = internal::AddOnImpl<FST, Data>;
  using D = Data;

  friend class StateIterator<MatcherFst<FST, FstMatcher, Name, Init, Data>>;
  friend class ArcIterator<MatcherFst<FST, FstMatcher, Name, Init, Data>>;

  MatcherFst() : ImplToExpandedFst<Impl>(std::make_shared<Impl>(FST(), Name)) {}

  explicit MatcherFst(const FST &fst, std::shared_ptr<Data> data = nullptr)
      : ImplToExpandedFst<Impl>(data ? CreateImpl(fst, Name, data)
                                     : CreateDataAndImpl(fst, Name)) {}

  explicit MatcherFst(const Fst<Arc> &fst)
      : ImplToExpandedFst<Impl>(CreateDataAndImpl(fst, Name)) {}

  // See Fst<>::Copy() for doc.
  MatcherFst(const MatcherFst<FST, FstMatcher, Name, Init, Data> &fst,
             bool safe = false)
      : ImplToExpandedFst<Impl>(fst, safe) {}

  // Get a copy of this MatcherFst. See Fst<>::Copy() for further doc.
  MatcherFst<FST, FstMatcher, Name, Init, Data> *Copy(
      bool safe = false) const override {
    return new MatcherFst<FST, FstMatcher, Name, Init, Data>(*this, safe);
  }

  // Read a MatcherFst from an input stream; return nullptr on error
  static MatcherFst<FST, M, Name, Init, Data> *Read(
      std::istream &strm, const FstReadOptions &opts) {
    auto *impl = Impl::Read(strm, opts);
    return impl ? new MatcherFst<FST, FstMatcher, Name, Init, Data>(
                      std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  // Read a MatcherFst from a file; return nullptr on error
  // Empty filename reads from standard input
  static MatcherFst<FST, FstMatcher, Name, Init, Data> *Read(
      const string &filename) {
    auto *impl = ImplToExpandedFst<Impl>::Read(filename);
    return impl ? new MatcherFst<FST, FstMatcher, Name, Init, Data>(
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
    return GetImpl()->InitStateIterator(data);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    return GetImpl()->InitArcIterator(s, data);
  }

  FstMatcher *InitMatcher(MatchType match_type) const override {
    return new FstMatcher(&GetFst(), match_type, GetSharedData(match_type));
  }

  const FST &GetFst() const { return GetImpl()->GetFst(); }

  const Data *GetAddOn() const { return GetImpl()->GetAddOn(); }

  std::shared_ptr<Data> GetSharedAddOn() const {
    return GetImpl()->GetSharedAddOn();
  }

  const MatcherData *GetData(MatchType match_type) const {
    const auto *data = GetAddOn();
    return match_type == MATCH_INPUT ? data->First() : data->Second();
  }

  std::shared_ptr<MatcherData> GetSharedData(MatchType match_type) const {
    const auto *data = GetAddOn();
    return match_type == MATCH_INPUT ? data->SharedFirst()
                                     : data->SharedSecond();
  }

 protected:
  using ImplToFst<Impl, ExpandedFst<Arc>>::GetImpl;

  static std::shared_ptr<Impl> CreateDataAndImpl(const FST &fst,
                                                 const string &name) {
    FstMatcher imatcher(fst, MATCH_INPUT);
    FstMatcher omatcher(fst, MATCH_OUTPUT);
    return CreateImpl(fst, name,
                      std::make_shared<Data>(imatcher.GetSharedData(),
                                             omatcher.GetSharedData()));
  }

  static std::shared_ptr<Impl> CreateDataAndImpl(const Fst<Arc> &fst,
                                                 const string &name) {
    FST result(fst);
    return CreateDataAndImpl(result, name);
  }

  static std::shared_ptr<Impl> CreateImpl(const FST &fst, const string &name,
                                          std::shared_ptr<Data> data) {
    auto impl = std::make_shared<Impl>(fst, name);
    impl->SetAddOn(data);
    Init init(&impl);
    return impl;
  }

  explicit MatcherFst(std::shared_ptr<Impl> impl)
      : ImplToExpandedFst<Impl>(impl) {}

 private:
  MatcherFst &operator=(const MatcherFst &) = delete;
};

// Specialization for MatcherFst.
template <class FST, class M, const char *Name, class Init>
class StateIterator<MatcherFst<FST, M, Name, Init>>
    : public StateIterator<FST> {
 public:
  explicit StateIterator(const MatcherFst<FST, M, Name, Init> &fst)
      : StateIterator<FST>(fst.GetImpl()->GetFst()) {}
};

// Specialization for MatcherFst.
template <class FST, class M, const char *Name, class Init>
class ArcIterator<MatcherFst<FST, M, Name, Init>> : public ArcIterator<FST> {
 public:
  using StateId = typename FST::Arc::StateId;

  ArcIterator(const MatcherFst<FST, M, Name, Init> &fst,
              typename FST::Arc::StateId s)
      : ArcIterator<FST>(fst.GetImpl()->GetFst(), s) {}
};

// Specialization for MatcherFst.
template <class F, class M, const char *Name, class Init>
class Matcher<MatcherFst<F, M, Name, Init>> {
 public:
  using FST = MatcherFst<F, M, Name, Init>;
  using Arc = typename F::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;

  Matcher(const FST &fst, MatchType match_type)
      : matcher_(fst.InitMatcher(match_type)) {}

  Matcher(const Matcher<FST> &matcher) : matcher_(matcher.matcher_->Copy()) {}

  Matcher<FST> *Copy() const { return new Matcher<FST>(*this); }

  MatchType Type(bool test) const { return matcher_->Type(test); }

  void SetState(StateId s) { matcher_->SetState(s); }

  bool Find(Label label) { return matcher_->Find(label); }

  bool Done() const { return matcher_->Done(); }

  const Arc &Value() const { return matcher_->Value(); }

  void Next() { matcher_->Next(); }

  uint64 Properties(uint64 props) const { return matcher_->Properties(props); }

  uint32 Flags() const { return matcher_->Flags(); }

 private:
  std::unique_ptr<M> matcher_;
};

// Specialization for MatcherFst.
template <class F, class M, const char *Name, class Init>
class LookAheadMatcher<MatcherFst<F, M, Name, Init>> {
 public:
  using FST = MatcherFst<F, M, Name, Init>;
  using Arc = typename F::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  LookAheadMatcher(const FST &fst, MatchType match_type)
      : matcher_(fst.InitMatcher(match_type)) {}

  LookAheadMatcher(const LookAheadMatcher<FST> &matcher, bool safe = false)
      : matcher_(matcher.matcher_->Copy(safe)) {}

  // General matcher methods.
  LookAheadMatcher<FST> *Copy(bool safe = false) const {
    return new LookAheadMatcher<FST>(*this, safe);
  }

  MatchType Type(bool test) const { return matcher_->Type(test); }

  void SetState(StateId s) { matcher_->SetState(s); }

  bool Find(Label label) { return matcher_->Find(label); }

  bool Done() const { return matcher_->Done(); }

  const Arc &Value() const { return matcher_->Value(); }

  void Next() { matcher_->Next(); }

  const FST &GetFst() const { return matcher_->GetFst(); }

  uint64 Properties(uint64 props) const { return matcher_->Properties(props); }

  uint32 Flags() const { return matcher_->Flags(); }

  bool LookAheadLabel(Label label) const {
    return matcher_->LookAheadLabel(label);
  }

  bool LookAheadFst(const Fst<Arc> &fst, StateId s) {
    return matcher_->LookAheadFst(fst, s);
  }

  Weight LookAheadWeight() const { return matcher_->LookAheadWeight(); }

  bool LookAheadPrefix(Arc *arc) const {
    return matcher_->LookAheadPrefix(arc);
  }

  void InitLookAheadFst(const Fst<Arc> &fst, bool copy = false) {
    matcher_->InitLookAheadFst(fst, copy);
  }

 private:
  std::unique_ptr<M> matcher_;
};

// Useful aliases when using StdArc.

extern const char arc_lookahead_fst_type[];

using StdArcLookAheadFst =
    MatcherFst<ConstFst<StdArc>,
               ArcLookAheadMatcher<SortedMatcher<ConstFst<StdArc>>>,
               arc_lookahead_fst_type>;

extern const char ilabel_lookahead_fst_type[];
extern const char olabel_lookahead_fst_type[];

constexpr auto ilabel_lookahead_flags =
    kInputLookAheadMatcher | kLookAheadWeight | kLookAheadPrefix |
    kLookAheadEpsilons | kLookAheadNonEpsilonPrefix;

constexpr auto olabel_lookahead_flags =
    kOutputLookAheadMatcher | kLookAheadWeight | kLookAheadPrefix |
    kLookAheadEpsilons | kLookAheadNonEpsilonPrefix;

using StdILabelLookAheadFst = MatcherFst<
    ConstFst<StdArc>,
    LabelLookAheadMatcher<SortedMatcher<ConstFst<StdArc>>,
                          ilabel_lookahead_flags, FastLogAccumulator<StdArc>>,
    ilabel_lookahead_fst_type, LabelLookAheadRelabeler<StdArc>>;

using StdOLabelLookAheadFst = MatcherFst<
    ConstFst<StdArc>,
    LabelLookAheadMatcher<SortedMatcher<ConstFst<StdArc>>,
                          olabel_lookahead_flags, FastLogAccumulator<StdArc>>,
    olabel_lookahead_fst_type, LabelLookAheadRelabeler<StdArc>>;

}  // namespace fst

#endif  // FST_MATCHER_FST_H_
