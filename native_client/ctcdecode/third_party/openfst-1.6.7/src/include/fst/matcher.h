// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes to allow matching labels leaving FST states.

#ifndef FST_MATCHER_H_
#define FST_MATCHER_H_

#include <algorithm>
#include <unordered_map>
#include <utility>

#include <fst/log.h>

#include <fst/mutable-fst.h>  // for all internal FST accessors.


namespace fst {

// Matchers find and iterate through requested labels at FST states. In the
// simplest form, these are just some associative map or search keyed on labels.
// More generally, they may implement matching special labels that represent
// sets of labels such as sigma (all), rho (rest), or phi (fail). The Matcher
// interface is:
//
// template <class F>
// class Matcher {
//  public:
//   using FST = F;
//   using Arc = typename FST::Arc;
//   using Label = typename Arc::Label;
//   using StateId = typename Arc::StateId;
//   using Weight = typename Arc::Weight;
//
//   // Required constructors. Note:
//   // -- the constructors that copy the FST arg are useful for
//   // letting the matcher manage the FST through copies
//   // (esp with 'safe' copies); e.g. ComposeFst depends on this.
//   // -- the constructor that does not copy is useful when the
//   // the FST is mutated during the lifetime of the matcher
//   // (o.w. the matcher would have its own unmutated deep copy).
//
//   // This makes a copy of the FST.
//   Matcher(const FST &fst, MatchType type);
//   // This doesn't copy the FST.
//   Matcher(const FST *fst, MatchType type);
//   // This makes a copy of the FST.
//   // See Copy() below.
//   Matcher(const Matcher &matcher, bool safe = false);
//
//   // If safe = true, the copy is thread-safe. See Fst<>::Copy() for
//   // further doc.
//   Matcher<FST> *Copy(bool safe = false) const override;
//
//   // Returns the match type that can be provided (depending on compatibility
//   of the input FST). It is either the requested match type, MATCH_NONE, or
//   MATCH_UNKNOWN. If test is false, a costly testing is avoided, but
//   MATCH_UNKNOWN may be returned. If test is true, a definite answer is
//   returned, but may involve more costly computation (e.g., visiting the FST).
//   MatchType Type(bool test) const override;
//
//   // Specifies the current state.
//   void SetState(StateId s) final;
//
//   // Finds matches to a label at the current state, returning true if a match
//   // found. kNoLabel matches any non-consuming transitions, e.g., epsilon
//   // transitions, which do not require a matching symbol.
//   bool Find(Label label) final;
//
//   // Iterator methods. Note that initially and after SetState() these have
//   undefined behavior until Find() is called.
//
//   bool Done() const final;
//
//   const Arc &Value() const final;
//
//   void Next() final;
//
//   // Returns final weight of a state.
//   Weight Final(StateId) const final;
//
//   // Indicates preference for being the side used for matching in
//   // composition. If the value is kRequirePriority, then it is
//   // mandatory that it be used. Calling this method without passing the
//   // current state of the matcher invalidates the state of the matcher.
//   ssize_t Priority(StateId s) final;
//
//   // This specifies the known FST properties as viewed from this matcher. It
//   // takes as argument the input FST's known properties.
//   uint64 Properties(uint64 props) const override;
//
//   // Returns matcher flags.
//   uint32 Flags() const override;
//
//   // Returns matcher FST.
//   const FST &GetFst() const override;
// };

// Basic matcher flags.

// Matcher needs to be used as the matching side in composition for
// at least one state (has kRequirePriority).
constexpr uint32 kRequireMatch = 0x00000001;

// Flags used for basic matchers (see also lookahead.h).
constexpr uint32 kMatcherFlags = kRequireMatch;

// Matcher priority that is mandatory.
constexpr ssize_t kRequirePriority = -1;

// Matcher interface, templated on the Arc definition; used for matcher
// specializations that are returned by the InitMatcher FST method.
template <class A>
class MatcherBase {
 public:
  using Arc = A;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  virtual ~MatcherBase() {}

  // Virtual interface.

  virtual MatcherBase<Arc> *Copy(bool safe = false) const = 0;
  virtual MatchType Type(bool) const = 0;
  virtual void SetState(StateId) = 0;
  virtual bool Find(Label) = 0;
  virtual bool Done() const = 0;
  virtual const Arc &Value() const = 0;
  virtual void Next() = 0;
  virtual const Fst<Arc> &GetFst() const = 0;
  virtual uint64 Properties(uint64) const = 0;

  // Trivial implementations that can be used by derived classes. Full
  // devirtualization is expected for any derived class marked final.
  virtual uint32 Flags() const { return 0; }

  virtual Weight Final(StateId s) const { return internal::Final(GetFst(), s); }

  virtual ssize_t Priority(StateId s) { return internal::NumArcs(GetFst(), s); }
};

// A matcher that expects sorted labels on the side to be matched.
// If match_type == MATCH_INPUT, epsilons match the implicit self-loop
// Arc(kNoLabel, 0, Weight::One(), current_state) as well as any
// actual epsilon transitions. If match_type == MATCH_OUTPUT, then
// Arc(0, kNoLabel, Weight::One(), current_state) is instead matched.
template <class F>
class SortedMatcher : public MatcherBase<typename F::Arc> {
 public:
  using FST = F;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using MatcherBase<Arc>::Flags;
  using MatcherBase<Arc>::Properties;

  // Labels >= binary_label will be searched for by binary search;
  // o.w. linear search is used.
  // This makes a copy of the FST.
  SortedMatcher(const FST &fst, MatchType match_type, Label binary_label = 1)
      : SortedMatcher(fst.Copy(), match_type, binary_label) {
    owned_fst_.reset(&fst_);
  }

  // Labels >= binary_label will be searched for by binary search;
  // o.w. linear search is used.
  // This doesn't copy the FST.
  SortedMatcher(const FST *fst, MatchType match_type, Label binary_label = 1)
      : fst_(*fst),
        state_(kNoStateId),
        aiter_(nullptr),
        match_type_(match_type),
        binary_label_(binary_label),
        match_label_(kNoLabel),
        narcs_(0),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId),
        error_(false),
        aiter_pool_(1) {
    switch (match_type_) {
      case MATCH_INPUT:
      case MATCH_NONE:
        break;
      case MATCH_OUTPUT:
        std::swap(loop_.ilabel, loop_.olabel);
        break;
      default:
        FSTERROR() << "SortedMatcher: Bad match type";
        match_type_ = MATCH_NONE;
        error_ = true;
    }
  }

  // This makes a copy of the FST.
  SortedMatcher(const SortedMatcher<FST> &matcher, bool safe = false)
      : owned_fst_(matcher.fst_.Copy(safe)),
        fst_(*owned_fst_),
        state_(kNoStateId),
        aiter_(nullptr),
        match_type_(matcher.match_type_),
        binary_label_(matcher.binary_label_),
        match_label_(kNoLabel),
        narcs_(0),
        loop_(matcher.loop_),
        error_(matcher.error_),
        aiter_pool_(1) {}

  ~SortedMatcher() override { Destroy(aiter_, &aiter_pool_); }

  SortedMatcher<FST> *Copy(bool safe = false) const override {
    return new SortedMatcher<FST>(*this, safe);
  }

  MatchType Type(bool test) const override {
    if (match_type_ == MATCH_NONE) return match_type_;
    const auto true_prop =
        match_type_ == MATCH_INPUT ? kILabelSorted : kOLabelSorted;
    const auto false_prop =
        match_type_ == MATCH_INPUT ? kNotILabelSorted : kNotOLabelSorted;
    const auto props = fst_.Properties(true_prop | false_prop, test);
    if (props & true_prop) {
      return match_type_;
    } else if (props & false_prop) {
      return MATCH_NONE;
    } else {
      return MATCH_UNKNOWN;
    }
  }

  void SetState(StateId s) final {
    if (state_ == s) return;
    state_ = s;
    if (match_type_ == MATCH_NONE) {
      FSTERROR() << "SortedMatcher: Bad match type";
      error_ = true;
    }
    Destroy(aiter_, &aiter_pool_);
    aiter_ = new (&aiter_pool_) ArcIterator<FST>(fst_, s);
    aiter_->SetFlags(kArcNoCache, kArcNoCache);
    narcs_ = internal::NumArcs(fst_, s);
    loop_.nextstate = s;
  }

  bool Find(Label match_label) final {
    exact_match_ = true;
    if (error_) {
      current_loop_ = false;
      match_label_ = kNoLabel;
      return false;
    }
    current_loop_ = match_label == 0;
    match_label_ = match_label == kNoLabel ? 0 : match_label;
    if (Search()) {
      return true;
    } else {
      return current_loop_;
    }
  }

  // Positions matcher to the first position where inserting match_label would
  // maintain the sort order.
  void LowerBound(Label label) {
    exact_match_ = false;
    current_loop_ = false;
    if (error_) {
      match_label_ = kNoLabel;
      return;
    }
    match_label_ = label;
    Search();
  }

  // After Find(), returns false if no more exact matches.
  // After LowerBound(), returns false if no more arcs.
  bool Done() const final {
    if (current_loop_) return false;
    if (aiter_->Done()) return true;
    if (!exact_match_) return false;
    aiter_->SetFlags(match_type_ == MATCH_INPUT ?
        kArcILabelValue : kArcOLabelValue,
        kArcValueFlags);
    return GetLabel() != match_label_;
  }

  const Arc &Value() const final {
    if (current_loop_) return loop_;
    aiter_->SetFlags(kArcValueFlags, kArcValueFlags);
    return aiter_->Value();
  }

  void Next() final {
    if (current_loop_) {
      current_loop_ = false;
    } else {
      aiter_->Next();
    }
  }

  Weight Final(StateId s) const final {
    return MatcherBase<Arc>::Final(s);
  }

  ssize_t Priority(StateId s) final {
    return MatcherBase<Arc>::Priority(s);
  }

  const FST &GetFst() const override { return fst_; }

  uint64 Properties(uint64 inprops) const override {
    return inprops | (error_ ? kError : 0);
  }

  size_t Position() const { return aiter_ ? aiter_->Position() : 0; }

 private:
  Label GetLabel() const {
    const auto &arc = aiter_->Value();
    return match_type_ == MATCH_INPUT ? arc.ilabel : arc.olabel;
  }

  bool BinarySearch();
  bool LinearSearch();
  bool Search();

  std::unique_ptr<const FST> owned_fst_;   // FST ptr if owned.
  const FST &fst_;           // FST for matching.
  StateId state_;            // Matcher state.
  ArcIterator<FST> *aiter_;  // Iterator for current state.
  MatchType match_type_;     // Type of match to perform.
  Label binary_label_;       // Least label for binary search.
  Label match_label_;        // Current label to be matched.
  size_t narcs_;             // Current state arc count.
  Arc loop_;                 // For non-consuming symbols.
  bool current_loop_;        // Current arc is the implicit loop.
  bool exact_match_;         // Exact match or lower bound?
  bool error_;               // Error encountered?
  MemoryPool<ArcIterator<FST>> aiter_pool_;  // Pool of arc iterators.
};

// Returns true iff match to match_label_. The arc iterator is positioned at the
// lower bound, that is, the first element greater than or equal to
// match_label_, or the end if all elements are less than match_label_.
template <class FST>
inline bool SortedMatcher<FST>::BinarySearch() {
  size_t low = 0;
  size_t high = narcs_;
  while (low < high) {
    const size_t mid = low + (high - low) / 2;
    aiter_->Seek(mid);
    if (GetLabel() < match_label_) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  aiter_->Seek(low);
  return low < narcs_ && GetLabel() == match_label_;
}

// Returns true iff match to match_label_, positioning arc iterator at lower
// bound.
template <class FST>
inline bool SortedMatcher<FST>::LinearSearch() {
  for (aiter_->Reset(); !aiter_->Done(); aiter_->Next()) {
    const auto label = GetLabel();
    if (label == match_label_) return true;
    if (label > match_label_) break;
  }
  return false;
}

// Returns true iff match to match_label_, positioning arc iterator at lower
// bound.
template <class FST>
inline bool SortedMatcher<FST>::Search() {
  aiter_->SetFlags(match_type_ == MATCH_INPUT ?
                   kArcILabelValue : kArcOLabelValue,
                   kArcValueFlags);
  if (match_label_ >= binary_label_) {
    return BinarySearch();
  } else {
    return LinearSearch();
  }
}

// A matcher that stores labels in a per-state hash table populated upon the
// first visit to that state. Sorting is not required. Treatment of
// epsilons are the same as with SortedMatcher.
template <class F>
class HashMatcher : public MatcherBase<typename F::Arc> {
 public:
  using FST = F;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using MatcherBase<Arc>::Flags;
  using MatcherBase<Arc>::Final;
  using MatcherBase<Arc>::Priority;

  // This makes a copy of the FST.
  HashMatcher(const FST &fst, MatchType match_type)
      : HashMatcher(fst.Copy(), match_type) {
    owned_fst_.reset(&fst_);
  }

  // This doesn't copy the FST.
  HashMatcher(const FST *fst, MatchType match_type)
      : fst_(*fst),
        state_(kNoStateId),
        match_type_(match_type),
        loop_(kNoLabel, 0, Weight::One(), kNoStateId),
        error_(false) {
    switch (match_type_) {
      case MATCH_INPUT:
      case MATCH_NONE:
        break;
      case MATCH_OUTPUT:
        std::swap(loop_.ilabel, loop_.olabel);
        break;
      default:
        FSTERROR() << "HashMatcher: Bad match type";
        match_type_ = MATCH_NONE;
        error_ = true;
    }
  }

  // This makes a copy of the FST.
  HashMatcher(const HashMatcher<FST> &matcher, bool safe = false)
      : owned_fst_(matcher.fst_.Copy(safe)),
        fst_(*owned_fst_),
        state_(kNoStateId),
        match_type_(matcher.match_type_),
        loop_(matcher.loop_),
        error_(matcher.error_) {}

  HashMatcher<FST> *Copy(bool safe = false) const override {
    return new HashMatcher<FST>(*this, safe);
  }

  // The argument is ignored as there are no relevant properties to test.
  MatchType Type(bool test) const override { return match_type_; }

  void SetState(StateId s) final;

  bool Find(Label label) final {
    current_loop_ = label == 0;
    if (label == 0) {
      Search(label);
      return true;
    }
    if (label == kNoLabel) label = 0;
    return Search(label);
  }

  bool Done() const final {
    if (current_loop_) return false;
    return label_it_ == label_end_;
  }

  const Arc &Value() const final {
    if (current_loop_) return loop_;
    aiter_->Seek(label_it_->second);
    return aiter_->Value();
  }

  void Next() final {
    if (current_loop_) {
      current_loop_ = false;
    } else {
      ++label_it_;
    }
  }

  const FST &GetFst() const override { return fst_; }

  uint64 Properties(uint64 inprops) const override {
    return inprops | (error_ ? kError : 0);
  }

 private:
  Label GetLabel() const {
    const auto &arc = aiter_->Value();
    return match_type_ == MATCH_INPUT ? arc.ilabel : arc.olabel;
  }

  bool Search(Label match_label);

  using LabelTable = std::unordered_multimap<Label, size_t>;
  using StateTable = std::unordered_map<StateId, LabelTable>;

  std::unique_ptr<const FST> owned_fst_;  // ptr to FST if owned.
  const FST &fst_;     // FST for matching.
  StateId state_;      // Matcher state.
  MatchType match_type_;
  Arc loop_;            // The implicit loop itself.
  bool current_loop_;   // Is the current arc the implicit loop?
  bool error_;          // Error encountered?
  std::unique_ptr<ArcIterator<FST>> aiter_;
  StateTable state_table_;   // Table from states to label table.
  LabelTable *label_table_;  // Pointer to current state's label table.
  typename LabelTable::iterator label_it_;   // Position for label.
  typename LabelTable::iterator label_end_;  // Position for last label + 1.
};

template <class FST>
void HashMatcher<FST>::SetState(typename FST::Arc::StateId s) {
  if (state_ == s) return;
  // Resets everything for the state.
  state_ = s;
  loop_.nextstate = state_;
  aiter_.reset(new ArcIterator<FST>(fst_, state_));
  if (match_type_ == MATCH_NONE) {
    FSTERROR() << "HashMatcher: Bad match type";
    error_ = true;
  }
  // Attempts to insert a new label table; if it already exists,
  // no additional work is done and we simply return.
  auto it_and_success = state_table_.emplace(state_, LabelTable());
  if (!it_and_success.second) return;
  // Otherwise, populate this new table.
  // Sets instance's pointer to the label table for this state.
  label_table_ = &(it_and_success.first->second);
  // Populates the label table.
  label_table_->reserve(internal::NumArcs(fst_, state_));
  const auto aiter_flags =
      (match_type_ == MATCH_INPUT ? kArcILabelValue : kArcOLabelValue) |
      kArcNoCache;
  aiter_->SetFlags(aiter_flags, kArcFlags);
  for (; !aiter_->Done(); aiter_->Next()) {
    label_table_->emplace(GetLabel(), aiter_->Position());
  }
  aiter_->SetFlags(kArcValueFlags, kArcValueFlags);
}

template <class FST>
inline bool HashMatcher<FST>::Search(typename FST::Arc::Label match_label) {
  auto range = label_table_->equal_range(match_label);
  if (range.first == range.second) return false;
  label_it_ = range.first;
  label_end_ = range.second;
  aiter_->Seek(label_it_->second);
  return true;
}

// Specifies whether we rewrite both the input and output sides during matching.
enum MatcherRewriteMode {
  MATCHER_REWRITE_AUTO = 0,  // Rewrites both sides iff acceptor.
  MATCHER_REWRITE_ALWAYS,
  MATCHER_REWRITE_NEVER
};

// For any requested label that doesn't match at a state, this matcher
// considers the *unique* transition that matches the label 'phi_label'
// (phi = 'fail'), and recursively looks for a match at its
// destination.  When 'phi_loop' is true, if no match is found but a
// phi self-loop is found, then the phi transition found is returned
// with the phi_label rewritten as the requested label (both sides if
// an acceptor, or if 'rewrite_both' is true and both input and output
// labels of the found transition are 'phi_label').  If 'phi_label' is
// kNoLabel, this special matching is not done.  PhiMatcher is
// templated itself on a matcher, which is used to perform the
// underlying matching.  By default, the underlying matcher is
// constructed by PhiMatcher. The user can instead pass in this
// object; in that case, PhiMatcher takes its ownership.
// Phi non-determinism not supported. No non-consuming symbols other
// than epsilon supported with the underlying template argument matcher.
template <class M>
class PhiMatcher : public MatcherBase<typename M::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST (w/o 'matcher' arg).
  PhiMatcher(const FST &fst, MatchType match_type, Label phi_label = kNoLabel,
             bool phi_loop = true,
             MatcherRewriteMode rewrite_mode = MATCHER_REWRITE_AUTO,
             M *matcher = nullptr)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        match_type_(match_type),
        phi_label_(phi_label),
        state_(kNoStateId),
        phi_loop_(phi_loop),
        error_(false) {
    if (match_type == MATCH_BOTH) {
      FSTERROR() << "PhiMatcher: Bad match type";
      match_type_ = MATCH_NONE;
      error_ = true;
    }
    if (rewrite_mode == MATCHER_REWRITE_AUTO) {
      rewrite_both_ = fst.Properties(kAcceptor, true);
    } else if (rewrite_mode == MATCHER_REWRITE_ALWAYS) {
      rewrite_both_ = true;
    } else {
      rewrite_both_ = false;
    }
  }

  // This doesn't copy the FST.
  PhiMatcher(const FST *fst, MatchType match_type, Label phi_label = kNoLabel,
             bool phi_loop = true,
             MatcherRewriteMode rewrite_mode = MATCHER_REWRITE_AUTO,
             M *matcher = nullptr)
      : PhiMatcher(*fst, match_type, phi_label, phi_loop, rewrite_mode,
                   matcher ? matcher : new M(fst, match_type)) { }


  // This makes a copy of the FST.
  PhiMatcher(const PhiMatcher<M> &matcher, bool safe = false)
      : matcher_(new M(*matcher.matcher_, safe)),
        match_type_(matcher.match_type_),
        phi_label_(matcher.phi_label_),
        rewrite_both_(matcher.rewrite_both_),
        state_(kNoStateId),
        phi_loop_(matcher.phi_loop_),
        error_(matcher.error_) {}

  PhiMatcher<M> *Copy(bool safe = false) const override {
    return new PhiMatcher<M>(*this, safe);
  }

  MatchType Type(bool test) const override { return matcher_->Type(test); }

  void SetState(StateId s) final {
    if (state_ == s) return;
    matcher_->SetState(s);
    state_ = s;
    has_phi_ = phi_label_ != kNoLabel;
  }

  bool Find(Label match_label) final;

  bool Done() const final { return matcher_->Done(); }

  const Arc &Value() const final {
    if ((phi_match_ == kNoLabel) && (phi_weight_ == Weight::One())) {
      return matcher_->Value();
    } else if (phi_match_ == 0) {  // Virtual epsilon loop.
      phi_arc_ = Arc(kNoLabel, 0, Weight::One(), state_);
      if (match_type_ == MATCH_OUTPUT) {
        std::swap(phi_arc_.ilabel, phi_arc_.olabel);
      }
      return phi_arc_;
    } else {
      phi_arc_ = matcher_->Value();
      phi_arc_.weight = Times(phi_weight_, phi_arc_.weight);
      if (phi_match_ != kNoLabel) {  // Phi loop match.
        if (rewrite_both_) {
          if (phi_arc_.ilabel == phi_label_) phi_arc_.ilabel = phi_match_;
          if (phi_arc_.olabel == phi_label_) phi_arc_.olabel = phi_match_;
        } else if (match_type_ == MATCH_INPUT) {
          phi_arc_.ilabel = phi_match_;
        } else {
          phi_arc_.olabel = phi_match_;
        }
      }
      return phi_arc_;
    }
  }

  void Next() final { matcher_->Next(); }

  Weight Final(StateId s) const final {
    auto weight = matcher_->Final(s);
    if (phi_label_ == kNoLabel || weight != Weight::Zero()) {
      return weight;
    }
    weight = Weight::One();
    matcher_->SetState(s);
    while (matcher_->Final(s) == Weight::Zero()) {
      if (!matcher_->Find(phi_label_ == 0 ? -1 : phi_label_)) break;
      weight = Times(weight, matcher_->Value().weight);
      if (s == matcher_->Value().nextstate) {
        return Weight::Zero();  // Does not follow phi self-loops.
      }
      s = matcher_->Value().nextstate;
      matcher_->SetState(s);
    }
    weight = Times(weight, matcher_->Final(s));
    return weight;
  }

  ssize_t Priority(StateId s) final {
    if (phi_label_ != kNoLabel) {
      matcher_->SetState(s);
      const bool has_phi = matcher_->Find(phi_label_ == 0 ? -1 : phi_label_);
      return has_phi ? kRequirePriority : matcher_->Priority(s);
    } else {
      return matcher_->Priority(s);
    }
  }

  const FST &GetFst() const override { return matcher_->GetFst(); }

  uint64 Properties(uint64 props) const override;

  uint32 Flags() const override {
    if (phi_label_ == kNoLabel || match_type_ == MATCH_NONE) {
      return matcher_->Flags();
    }
    return matcher_->Flags() | kRequireMatch;
  }

  Label PhiLabel() const { return phi_label_; }

 private:
  mutable std::unique_ptr<M> matcher_;
  MatchType match_type_;  // Type of match requested.
  Label phi_label_;       // Label that represents the phi transition.
  bool rewrite_both_;     // Rewrite both sides when both are phi_label_?
  bool has_phi_;          // Are there possibly phis at the current state?
  Label phi_match_;       // Current label that matches phi loop.
  mutable Arc phi_arc_;   // Arc to return.
  StateId state_;         // Matcher state.
  Weight phi_weight_;     // Product of the weights of phi transitions taken.
  bool phi_loop_;         // When true, phi self-loop are allowed and treated
                          // as rho (required for Aho-Corasick).
  bool error_;            // Error encountered?

  PhiMatcher &operator=(const PhiMatcher &) = delete;
};

template <class M>
inline bool PhiMatcher<M>::Find(Label label) {
  if (label == phi_label_ && phi_label_ != kNoLabel && phi_label_ != 0) {
    FSTERROR() << "PhiMatcher::Find: bad label (phi): " << phi_label_;
    error_ = true;
    return false;
  }
  matcher_->SetState(state_);
  phi_match_ = kNoLabel;
  phi_weight_ = Weight::One();
  // If phi_label_ == 0, there are no more true epsilon arcs.
  if (phi_label_ == 0) {
    if (label == kNoLabel) {
      return false;
    }
    if (label == 0) {  // but a virtual epsilon loop needs to be returned.
      if (!matcher_->Find(kNoLabel)) {
        return matcher_->Find(0);
      } else {
        phi_match_ = 0;
        return true;
      }
    }
  }
  if (!has_phi_ || label == 0 || label == kNoLabel) {
    return matcher_->Find(label);
  }
  auto s = state_;
  while (!matcher_->Find(label)) {
    // Look for phi transition (if phi_label_ == 0, we need to look
    // for -1 to avoid getting the virtual self-loop)
    if (!matcher_->Find(phi_label_ == 0 ? -1 : phi_label_)) return false;
    if (phi_loop_ && matcher_->Value().nextstate == s) {
      phi_match_ = label;
      return true;
    }
    phi_weight_ = Times(phi_weight_, matcher_->Value().weight);
    s = matcher_->Value().nextstate;
    matcher_->Next();
    if (!matcher_->Done()) {
      FSTERROR() << "PhiMatcher: Phi non-determinism not supported";
      error_ = true;
    }
    matcher_->SetState(s);
  }
  return true;
}

template <class M>
inline uint64 PhiMatcher<M>::Properties(uint64 inprops) const {
  auto outprops = matcher_->Properties(inprops);
  if (error_) outprops |= kError;
  if (match_type_ == MATCH_NONE) {
    return outprops;
  } else if (match_type_ == MATCH_INPUT) {
    if (phi_label_ == 0) {
      outprops &= ~kEpsilons | ~kIEpsilons | ~kOEpsilons;
      outprops |= kNoEpsilons | kNoIEpsilons;
    }
    if (rewrite_both_) {
      return outprops &
             ~(kODeterministic | kNonODeterministic | kString | kILabelSorted |
               kNotILabelSorted | kOLabelSorted | kNotOLabelSorted);
    } else {
      return outprops &
             ~(kODeterministic | kAcceptor | kString | kILabelSorted |
               kNotILabelSorted | kOLabelSorted | kNotOLabelSorted);
    }
  } else if (match_type_ == MATCH_OUTPUT) {
    if (phi_label_ == 0) {
      outprops &= ~kEpsilons | ~kIEpsilons | ~kOEpsilons;
      outprops |= kNoEpsilons | kNoOEpsilons;
    }
    if (rewrite_both_) {
      return outprops &
             ~(kIDeterministic | kNonIDeterministic | kString | kILabelSorted |
               kNotILabelSorted | kOLabelSorted | kNotOLabelSorted);
    } else {
      return outprops &
             ~(kIDeterministic | kAcceptor | kString | kILabelSorted |
               kNotILabelSorted | kOLabelSorted | kNotOLabelSorted);
    }
  } else {
    // Shouldn't ever get here.
    FSTERROR() << "PhiMatcher: Bad match type: " << match_type_;
    return 0;
  }
}

// For any requested label that doesn't match at a state, this matcher
// considers all transitions that match the label 'rho_label' (rho =
// 'rest').  Each such rho transition found is returned with the
// rho_label rewritten as the requested label (both sides if an
// acceptor, or if 'rewrite_both' is true and both input and output
// labels of the found transition are 'rho_label').  If 'rho_label' is
// kNoLabel, this special matching is not done.  RhoMatcher is
// templated itself on a matcher, which is used to perform the
// underlying matching.  By default, the underlying matcher is
// constructed by RhoMatcher.  The user can instead pass in this
// object; in that case, RhoMatcher takes its ownership.
// No non-consuming symbols other than epsilon supported with
// the underlying template argument matcher.
template <class M>
class RhoMatcher : public MatcherBase<typename M::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST (w/o 'matcher' arg).
  RhoMatcher(const FST &fst, MatchType match_type, Label rho_label = kNoLabel,
             MatcherRewriteMode rewrite_mode = MATCHER_REWRITE_AUTO,
             M *matcher = nullptr)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        match_type_(match_type),
        rho_label_(rho_label),
        error_(false),
        state_(kNoStateId),
        has_rho_(false) {
    if (match_type == MATCH_BOTH) {
      FSTERROR() << "RhoMatcher: Bad match type";
      match_type_ = MATCH_NONE;
      error_ = true;
    }
    if (rho_label == 0) {
      FSTERROR() << "RhoMatcher: 0 cannot be used as rho_label";
      rho_label_ = kNoLabel;
      error_ = true;
    }
    if (rewrite_mode == MATCHER_REWRITE_AUTO) {
      rewrite_both_ = fst.Properties(kAcceptor, true);
    } else if (rewrite_mode == MATCHER_REWRITE_ALWAYS) {
      rewrite_both_ = true;
    } else {
      rewrite_both_ = false;
    }
  }

  // This doesn't copy the FST.
  RhoMatcher(const FST *fst, MatchType match_type, Label rho_label = kNoLabel,
             MatcherRewriteMode rewrite_mode = MATCHER_REWRITE_AUTO,
             M *matcher = nullptr)
      : RhoMatcher(*fst, match_type, rho_label, rewrite_mode,
                   matcher ? matcher : new M(fst, match_type)) { }

  // This makes a copy of the FST.
  RhoMatcher(const RhoMatcher<M> &matcher, bool safe = false)
      : matcher_(new M(*matcher.matcher_, safe)),
        match_type_(matcher.match_type_),
        rho_label_(matcher.rho_label_),
        rewrite_both_(matcher.rewrite_both_),
        error_(matcher.error_),
        state_(kNoStateId),
        has_rho_(false) {}

  RhoMatcher<M> *Copy(bool safe = false) const override {
    return new RhoMatcher<M>(*this, safe);
  }

  MatchType Type(bool test) const override { return matcher_->Type(test); }

  void SetState(StateId s) final {
    if (state_ == s) return;
    state_ = s;
    matcher_->SetState(s);
    has_rho_ = rho_label_ != kNoLabel;
  }

  bool Find(Label label) final {
    if (label == rho_label_ && rho_label_ != kNoLabel) {
      FSTERROR() << "RhoMatcher::Find: bad label (rho)";
      error_ = true;
      return false;
    }
    if (matcher_->Find(label)) {
      rho_match_ = kNoLabel;
      return true;
    } else if (has_rho_ && label != 0 && label != kNoLabel &&
               (has_rho_ = matcher_->Find(rho_label_))) {
      rho_match_ = label;
      return true;
    } else {
      return false;
    }
  }

  bool Done() const final { return matcher_->Done(); }

  const Arc &Value() const final {
    if (rho_match_ == kNoLabel) {
      return matcher_->Value();
    } else {
      rho_arc_ = matcher_->Value();
      if (rewrite_both_) {
        if (rho_arc_.ilabel == rho_label_) rho_arc_.ilabel = rho_match_;
        if (rho_arc_.olabel == rho_label_) rho_arc_.olabel = rho_match_;
      } else if (match_type_ == MATCH_INPUT) {
        rho_arc_.ilabel = rho_match_;
      } else {
        rho_arc_.olabel = rho_match_;
      }
      return rho_arc_;
    }
  }

  void Next() final { matcher_->Next(); }

  Weight Final(StateId s) const final { return matcher_->Final(s); }

  ssize_t Priority(StateId s) final {
    state_ = s;
    matcher_->SetState(s);
    has_rho_ = matcher_->Find(rho_label_);
    if (has_rho_) {
      return kRequirePriority;
    } else {
      return matcher_->Priority(s);
    }
  }

  const FST &GetFst() const override { return matcher_->GetFst(); }

  uint64 Properties(uint64 props) const override;

  uint32 Flags() const override {
    if (rho_label_ == kNoLabel || match_type_ == MATCH_NONE) {
      return matcher_->Flags();
    }
    return matcher_->Flags() | kRequireMatch;
  }

  Label RhoLabel() const { return rho_label_; }

 private:
  std::unique_ptr<M> matcher_;
  MatchType match_type_;  // Type of match requested.
  Label rho_label_;       // Label that represents the rho transition
  bool rewrite_both_;     // Rewrite both sides when both are rho_label_?
  Label rho_match_;       // Current label that matches rho transition.
  mutable Arc rho_arc_;   // Arc to return when rho match.
  bool error_;            // Error encountered?
  StateId state_;         // Matcher state.
  bool has_rho_;          // Are there possibly rhos at the current state?
};

template <class M>
inline uint64 RhoMatcher<M>::Properties(uint64 inprops) const {
  auto outprops = matcher_->Properties(inprops);
  if (error_) outprops |= kError;
  if (match_type_ == MATCH_NONE) {
    return outprops;
  } else if (match_type_ == MATCH_INPUT) {
    if (rewrite_both_) {
      return outprops &
             ~(kODeterministic | kNonODeterministic | kString | kILabelSorted |
               kNotILabelSorted | kOLabelSorted | kNotOLabelSorted);
    } else {
      return outprops &
             ~(kODeterministic | kAcceptor | kString | kILabelSorted |
               kNotILabelSorted);
    }
  } else if (match_type_ == MATCH_OUTPUT) {
    if (rewrite_both_) {
      return outprops &
             ~(kIDeterministic | kNonIDeterministic | kString | kILabelSorted |
               kNotILabelSorted | kOLabelSorted | kNotOLabelSorted);
    } else {
      return outprops &
             ~(kIDeterministic | kAcceptor | kString | kOLabelSorted |
               kNotOLabelSorted);
    }
  } else {
    // Shouldn't ever get here.
    FSTERROR() << "RhoMatcher: Bad match type: " << match_type_;
    return 0;
  }
}

// For any requested label, this matcher considers all transitions
// that match the label 'sigma_label' (sigma = "any"), and this in
// additions to transitions with the requested label.  Each such sigma
// transition found is returned with the sigma_label rewritten as the
// requested label (both sides if an acceptor, or if 'rewrite_both' is
// true and both input and output labels of the found transition are
// 'sigma_label').  If 'sigma_label' is kNoLabel, this special
// matching is not done.  SigmaMatcher is templated itself on a
// matcher, which is used to perform the underlying matching.  By
// default, the underlying matcher is constructed by SigmaMatcher.
// The user can instead pass in this object; in that case,
// SigmaMatcher takes its ownership.  No non-consuming symbols other
// than epsilon supported with the underlying template argument matcher.
template <class M>
class SigmaMatcher : public MatcherBase<typename M::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST (w/o 'matcher' arg).
  SigmaMatcher(const FST &fst, MatchType match_type,
               Label sigma_label = kNoLabel,
               MatcherRewriteMode rewrite_mode = MATCHER_REWRITE_AUTO,
               M *matcher = nullptr)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        match_type_(match_type),
        sigma_label_(sigma_label),
        error_(false),
        state_(kNoStateId) {
    if (match_type == MATCH_BOTH) {
      FSTERROR() << "SigmaMatcher: Bad match type";
      match_type_ = MATCH_NONE;
      error_ = true;
    }
    if (sigma_label == 0) {
      FSTERROR() << "SigmaMatcher: 0 cannot be used as sigma_label";
      sigma_label_ = kNoLabel;
      error_ = true;
    }
    if (rewrite_mode == MATCHER_REWRITE_AUTO) {
      rewrite_both_ = fst.Properties(kAcceptor, true);
    } else if (rewrite_mode == MATCHER_REWRITE_ALWAYS) {
      rewrite_both_ = true;
    } else {
      rewrite_both_ = false;
    }
  }

  // This doesn't copy the FST.
  SigmaMatcher(const FST *fst, MatchType match_type,
               Label sigma_label = kNoLabel,
               MatcherRewriteMode rewrite_mode = MATCHER_REWRITE_AUTO,
             M *matcher = nullptr)
      : SigmaMatcher(*fst, match_type, sigma_label, rewrite_mode,
                     matcher ? matcher : new M(fst, match_type)) { }

  // This makes a copy of the FST.
  SigmaMatcher(const SigmaMatcher<M> &matcher, bool safe = false)
      : matcher_(new M(*matcher.matcher_, safe)),
        match_type_(matcher.match_type_),
        sigma_label_(matcher.sigma_label_),
        rewrite_both_(matcher.rewrite_both_),
        error_(matcher.error_),
        state_(kNoStateId) {}

  SigmaMatcher<M> *Copy(bool safe = false) const override {
    return new SigmaMatcher<M>(*this, safe);
  }

  MatchType Type(bool test) const override { return matcher_->Type(test); }

  void SetState(StateId s) final {
    if (state_ == s) return;
    state_ = s;
    matcher_->SetState(s);
    has_sigma_ =
        (sigma_label_ != kNoLabel) ? matcher_->Find(sigma_label_) : false;
  }

  bool Find(Label match_label) final {
    match_label_ = match_label;
    if (match_label == sigma_label_ && sigma_label_ != kNoLabel) {
      FSTERROR() << "SigmaMatcher::Find: bad label (sigma)";
      error_ = true;
      return false;
    }
    if (matcher_->Find(match_label)) {
      sigma_match_ = kNoLabel;
      return true;
    } else if (has_sigma_ && match_label != 0 && match_label != kNoLabel &&
               matcher_->Find(sigma_label_)) {
      sigma_match_ = match_label;
      return true;
    } else {
      return false;
    }
  }

  bool Done() const final { return matcher_->Done(); }

  const Arc &Value() const final {
    if (sigma_match_ == kNoLabel) {
      return matcher_->Value();
    } else {
      sigma_arc_ = matcher_->Value();
      if (rewrite_both_) {
        if (sigma_arc_.ilabel == sigma_label_) sigma_arc_.ilabel = sigma_match_;
        if (sigma_arc_.olabel == sigma_label_) sigma_arc_.olabel = sigma_match_;
      } else if (match_type_ == MATCH_INPUT) {
        sigma_arc_.ilabel = sigma_match_;
      } else {
        sigma_arc_.olabel = sigma_match_;
      }
      return sigma_arc_;
    }
  }

  void Next() final {
    matcher_->Next();
    if (matcher_->Done() && has_sigma_ && (sigma_match_ == kNoLabel) &&
        (match_label_ > 0)) {
      matcher_->Find(sigma_label_);
      sigma_match_ = match_label_;
    }
  }

  Weight Final(StateId s) const final { return matcher_->Final(s); }

  ssize_t Priority(StateId s) final {
    if (sigma_label_ != kNoLabel) {
      SetState(s);
      return has_sigma_ ? kRequirePriority : matcher_->Priority(s);
    } else {
      return matcher_->Priority(s);
    }
  }

  const FST &GetFst() const override { return matcher_->GetFst(); }

  uint64 Properties(uint64 props) const override;

  uint32 Flags() const override {
    if (sigma_label_ == kNoLabel || match_type_ == MATCH_NONE) {
      return matcher_->Flags();
    }
    return matcher_->Flags() | kRequireMatch;
  }

  Label SigmaLabel() const { return sigma_label_; }

 private:
  std::unique_ptr<M> matcher_;
  MatchType match_type_;   // Type of match requested.
  Label sigma_label_;      // Label that represents the sigma transition.
  bool rewrite_both_;      // Rewrite both sides when both are sigma_label_?
  bool has_sigma_;         // Are there sigmas at the current state?
  Label sigma_match_;      // Current label that matches sigma transition.
  mutable Arc sigma_arc_;  // Arc to return when sigma match.
  Label match_label_;      // Label being matched.
  bool error_;             // Error encountered?
  StateId state_;          // Matcher state.
};

template <class M>
inline uint64 SigmaMatcher<M>::Properties(uint64 inprops) const {
  auto outprops = matcher_->Properties(inprops);
  if (error_) outprops |= kError;
  if (match_type_ == MATCH_NONE) {
    return outprops;
  } else if (rewrite_both_) {
    return outprops &
           ~(kIDeterministic | kNonIDeterministic | kODeterministic |
             kNonODeterministic | kILabelSorted | kNotILabelSorted |
             kOLabelSorted | kNotOLabelSorted | kString);
  } else if (match_type_ == MATCH_INPUT) {
    return outprops &
           ~(kIDeterministic | kNonIDeterministic | kODeterministic |
             kNonODeterministic | kILabelSorted | kNotILabelSorted | kString |
             kAcceptor);
  } else if (match_type_ == MATCH_OUTPUT) {
    return outprops &
           ~(kIDeterministic | kNonIDeterministic | kODeterministic |
             kNonODeterministic | kOLabelSorted | kNotOLabelSorted | kString |
             kAcceptor);
  } else {
    // Shouldn't ever get here.
    FSTERROR() << "SigmaMatcher: Bad match type: " << match_type_;
    return 0;
  }
}

// Flags for MultiEpsMatcher.

// Return multi-epsilon arcs for Find(kNoLabel).
const uint32 kMultiEpsList = 0x00000001;

// Return a kNolabel loop for Find(multi_eps).
const uint32 kMultiEpsLoop = 0x00000002;

// MultiEpsMatcher: allows treating multiple non-0 labels as
// non-consuming labels in addition to 0 that is always
// non-consuming. Precise behavior controlled by 'flags' argument. By
// default, the underlying matcher is constructed by
// MultiEpsMatcher. The user can instead pass in this object; in that
// case, MultiEpsMatcher takes its ownership iff 'own_matcher' is
// true.
template <class M>
class MultiEpsMatcher {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST (w/o 'matcher' arg).
  MultiEpsMatcher(const FST &fst, MatchType match_type,
                  uint32 flags = (kMultiEpsLoop | kMultiEpsList),
                  M *matcher = nullptr, bool own_matcher = true)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        flags_(flags),
        own_matcher_(matcher ? own_matcher : true) {
    Init(match_type);
  }

  // This doesn't copy the FST.
  MultiEpsMatcher(const FST *fst, MatchType match_type,
                  uint32 flags = (kMultiEpsLoop | kMultiEpsList),
                  M *matcher = nullptr, bool own_matcher = true)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        flags_(flags),
        own_matcher_(matcher ? own_matcher : true) {
    Init(match_type);
  }

  // This makes a copy of the FST.
  MultiEpsMatcher(const MultiEpsMatcher<M> &matcher, bool safe = false)
      : matcher_(new M(*matcher.matcher_, safe)),
        flags_(matcher.flags_),
        own_matcher_(true),
        multi_eps_labels_(matcher.multi_eps_labels_),
        loop_(matcher.loop_) {
    loop_.nextstate = kNoStateId;
  }

  ~MultiEpsMatcher() {
    if (own_matcher_) delete matcher_;
  }

  MultiEpsMatcher<M> *Copy(bool safe = false) const {
    return new MultiEpsMatcher<M>(*this, safe);
  }

  MatchType Type(bool test) const { return matcher_->Type(test); }

  void SetState(StateId state) {
    matcher_->SetState(state);
    loop_.nextstate = state;
  }

  bool Find(Label label);

  bool Done() const { return done_; }

  const Arc &Value() const { return current_loop_ ? loop_ : matcher_->Value(); }

  void Next() {
    if (!current_loop_) {
      matcher_->Next();
      done_ = matcher_->Done();
      if (done_ && multi_eps_iter_ != multi_eps_labels_.End()) {
        ++multi_eps_iter_;
        while ((multi_eps_iter_ != multi_eps_labels_.End()) &&
               !matcher_->Find(*multi_eps_iter_)) {
          ++multi_eps_iter_;
        }
        if (multi_eps_iter_ != multi_eps_labels_.End()) {
          done_ = false;
        } else {
          done_ = !matcher_->Find(kNoLabel);
        }
      }
    } else {
      done_ = true;
    }
  }

  const FST &GetFst() const { return matcher_->GetFst(); }

  uint64 Properties(uint64 props) const { return matcher_->Properties(props); }

  const M *GetMatcher() const { return matcher_; }

  Weight Final(StateId s) const { return matcher_->Final(s); }

  uint32 Flags() const { return matcher_->Flags(); }

  ssize_t Priority(StateId s) { return matcher_->Priority(s); }

  void AddMultiEpsLabel(Label label) {
    if (label == 0) {
      FSTERROR() << "MultiEpsMatcher: Bad multi-eps label: 0";
    } else {
      multi_eps_labels_.Insert(label);
    }
  }

  void RemoveMultiEpsLabel(Label label) {
    if (label == 0) {
      FSTERROR() << "MultiEpsMatcher: Bad multi-eps label: 0";
    } else {
      multi_eps_labels_.Erase(label);
    }
  }

  void ClearMultiEpsLabels() { multi_eps_labels_.Clear(); }

 private:
  void Init(MatchType match_type) {
    if (match_type == MATCH_INPUT) {
      loop_.ilabel = kNoLabel;
      loop_.olabel = 0;
    } else {
      loop_.ilabel = 0;
      loop_.olabel = kNoLabel;
    }
    loop_.weight = Weight::One();
    loop_.nextstate = kNoStateId;
  }

  M *matcher_;
  uint32 flags_;
  bool own_matcher_;  // Does this class delete the matcher?

  // Multi-eps label set.
  CompactSet<Label, kNoLabel> multi_eps_labels_;
  typename CompactSet<Label, kNoLabel>::const_iterator multi_eps_iter_;

  bool current_loop_;  // Current arc is the implicit loop?
  mutable Arc loop_;   // For non-consuming symbols.
  bool done_;          // Matching done?

  MultiEpsMatcher &operator=(const MultiEpsMatcher &) = delete;
};

template <class M>
inline bool MultiEpsMatcher<M>::Find(Label label) {
  multi_eps_iter_ = multi_eps_labels_.End();
  current_loop_ = false;
  bool ret;
  if (label == 0) {
    ret = matcher_->Find(0);
  } else if (label == kNoLabel) {
    if (flags_ & kMultiEpsList) {
      // Returns all non-consuming arcs (including epsilon).
      multi_eps_iter_ = multi_eps_labels_.Begin();
      while ((multi_eps_iter_ != multi_eps_labels_.End()) &&
             !matcher_->Find(*multi_eps_iter_)) {
        ++multi_eps_iter_;
      }
      if (multi_eps_iter_ != multi_eps_labels_.End()) {
        ret = true;
      } else {
        ret = matcher_->Find(kNoLabel);
      }
    } else {
      // Returns all epsilon arcs.
      ret = matcher_->Find(kNoLabel);
    }
  } else if ((flags_ & kMultiEpsLoop) &&
             multi_eps_labels_.Find(label) != multi_eps_labels_.End()) {
    // Returns implicit loop.
    current_loop_ = true;
    ret = true;
  } else {
    ret = matcher_->Find(label);
  }
  done_ = !ret;
  return ret;
}

// This class discards any implicit matches (e.g., the implicit epsilon
// self-loops in the SortedMatcher). Matchers are most often used in
// composition/intersection where the implicit matches are needed
// e.g. for epsilon processing. However, if a matcher is simply being
// used to look-up explicit label matches, this class saves the user
// from having to check for and discard the unwanted implicit matches
// themselves.
template <class M>
class ExplicitMatcher : public MatcherBase<typename M::Arc> {
 public:
  using FST = typename M::FST;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST.
  ExplicitMatcher(const FST &fst, MatchType match_type, M *matcher = nullptr)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        match_type_(match_type),
        error_(false) {}

  // This doesn't copy the FST.
  ExplicitMatcher(const FST *fst, MatchType match_type, M *matcher = nullptr)
      : matcher_(matcher ? matcher : new M(fst, match_type)),
        match_type_(match_type),
        error_(false) {}

  // This makes a copy of the FST.
  ExplicitMatcher(const ExplicitMatcher<M> &matcher, bool safe = false)
      : matcher_(new M(*matcher.matcher_, safe)),
        match_type_(matcher.match_type_),
        error_(matcher.error_) {}

  ExplicitMatcher<M> *Copy(bool safe = false) const override {
    return new ExplicitMatcher<M>(*this, safe);
  }

  MatchType Type(bool test) const override { return matcher_->Type(test); }

  void SetState(StateId s) final { matcher_->SetState(s); }

  bool Find(Label label) final {
    matcher_->Find(label);
    CheckArc();
    return !Done();
  }

  bool Done() const final { return matcher_->Done(); }

  const Arc &Value() const final { return matcher_->Value(); }

  void Next() final {
    matcher_->Next();
    CheckArc();
  }

  Weight Final(StateId s) const final { return matcher_->Final(s); }

  ssize_t Priority(StateId s) final { return matcher_->Priority(s); }

  const FST &GetFst() const final { return matcher_->GetFst(); }

  uint64 Properties(uint64 inprops) const override {
    return matcher_->Properties(inprops);
  }

  const M *GetMatcher() const { return matcher_.get(); }

  uint32 Flags() const override { return matcher_->Flags(); }

 private:
  // Checks current arc if available and explicit. If not available, stops. If
  // not explicit, checks next ones.
  void CheckArc() {
    for (; !matcher_->Done(); matcher_->Next()) {
      const auto label = match_type_ == MATCH_INPUT ? matcher_->Value().ilabel
                                                    : matcher_->Value().olabel;
      if (label != kNoLabel) return;
    }
  }

  std::unique_ptr<M> matcher_;
  MatchType match_type_;  // Type of match requested.
  bool error_;            // Error encountered?
};

// Generic matcher, templated on the FST definition.
//
// Here is a typical use:
//
//   Matcher<StdFst> matcher(fst, MATCH_INPUT);
//   matcher.SetState(state);
//   if (matcher.Find(label))
//     for (; !matcher.Done(); matcher.Next()) {
//       auto &arc = matcher.Value();
//       ...
//     }
template <class F>
class Matcher {
 public:
  using FST = F;
  using Arc = typename F::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  // This makes a copy of the FST.
  Matcher(const FST &fst, MatchType match_type)
      : owned_fst_(fst.Copy()),
        base_(owned_fst_->InitMatcher(match_type)) {
    if (!base_) base_.reset(new SortedMatcher<FST>(owned_fst_.get(),
                                                   match_type));
  }

  // This doesn't copy the FST.
  Matcher(const FST *fst, MatchType match_type)
      : base_(fst->InitMatcher(match_type)) {
    if (!base_) base_.reset(new SortedMatcher<FST>(fst, match_type));
  }

  // This makes a copy of the FST.
  Matcher(const Matcher<FST> &matcher, bool safe = false)
      : base_(matcher.base_->Copy(safe)) { }

  // Takes ownership of the provided matcher.
  explicit Matcher(MatcherBase<Arc> *base_matcher)
      : base_(base_matcher) { }

  Matcher<FST> *Copy(bool safe = false) const {
    return new Matcher<FST>(*this, safe);
  }

  MatchType Type(bool test) const { return base_->Type(test); }

  void SetState(StateId s) { base_->SetState(s); }

  bool Find(Label label) { return base_->Find(label); }

  bool Done() const { return base_->Done(); }

  const Arc &Value() const { return base_->Value(); }

  void Next() { base_->Next(); }

  const FST &GetFst() const {
    return static_cast<const FST &>(base_->GetFst());
  }

  uint64 Properties(uint64 props) const { return base_->Properties(props); }

  Weight Final(StateId s) const { return base_->Final(s); }

  uint32 Flags() const { return base_->Flags() & kMatcherFlags; }

  ssize_t Priority(StateId s) { return base_->Priority(s); }

 private:
  std::unique_ptr<const FST> owned_fst_;
  std::unique_ptr<MatcherBase<Arc>> base_;
};

}  // namespace fst

#endif  // FST_MATCHER_H_
