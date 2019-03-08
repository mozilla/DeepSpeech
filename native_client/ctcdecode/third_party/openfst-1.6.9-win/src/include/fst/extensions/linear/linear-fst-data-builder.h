// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_LINEAR_LINEAR_FST_DATA_BUILDER_H_
#define FST_EXTENSIONS_LINEAR_LINEAR_FST_DATA_BUILDER_H_

#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

#include <fst/compat.h>
#include <fst/log.h>
#include <fst/fst.h>
#include <fst/symbol-table.h>
#include <fst/util.h>

#include <fst/extensions/linear/linear-fst-data.h>

namespace fst {

// Forward declaration
template <class A>
class FeatureGroupBuilder;

// For logging purposes
inline string TranslateLabel(int64_t label, const SymbolTable *syms);
template <class Iterator>
string JoinLabels(Iterator begin, Iterator end, const SymbolTable *syms);
template <class Label>
string JoinLabels(const std::vector<Label> &labels, const SymbolTable *syms);

// Guesses the appropriate boundary label (start- or end-of-sentence)
// for all labels equal to `boundary` and modifies the `sequence`
// in-place. Returns the number of positions that are still uncertain.
template <class A>
typename A::Label GuessStartOrEnd(std::vector<typename A::Label> *sequence,
                                  typename A::Label boundary);

// Builds a `LinearFstData` object by adding words and feature
// weights. A few conventions:
//
// - Input labels forms a dense non-empty range from 1 to `MaxInputLabel()`.
// - Feature labels, output labels are > 0.
// - Being a discriminative linear model, it only makes sense to use tropical
// semirings.
template <class A>
class LinearFstDataBuilder {
 public:
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;

  // Constructs a builder with associated symbol tables for diagonstic
  // output. Each of these symbol tables may also be nullptr.
  explicit LinearFstDataBuilder(const SymbolTable *isyms = nullptr,
                                const SymbolTable *fsyms = nullptr,
                                const SymbolTable *osyms = nullptr)
      : error_(false),
        max_future_size_(0),
        max_input_label_(1),
        isyms_(isyms),
        fsyms_(fsyms),
        osyms_(osyms) {}

  // Tests whether the builder has encountered any error. No operation
  // is valid if the builder is already at error state. All other
  // public methods should check this before any actual operations.
  bool Error() const { return error_; }

  // Adds a word and its feature labels to the vocabulary; this
  // version allows the word to have any output label. Returns true
  // iff the word is added.
  //
  // This may fail if the word is added twice or if the feature labels
  // are non-positive.
  bool AddWord(Label word, const std::vector<Label> &features);

  // Adds a word and its feature labels to the vocabulary; this
  // version puts constraint on possible output labels the word can
  // have. `possible_output` must not be empty. Returns true iff the
  // word is added.
  //
  // In addition to the reasons above in the two-parameter version,
  // this may also fail if `possible_output` is empty or any output
  // label in it is non-positive.
  bool AddWord(Label word, const std::vector<Label> &word_features,
               const std::vector<Label> &possible_output);

  // Creates a new feature group with specified future size (size of
  // the look-ahead window), returns the group id to be used for
  // adding actual feature weights or a negative number when called at
  // error state.
  //
  // This does not fail unless called at error state.
  int AddGroup(size_t future_size);

  // Adds an instance of feature weight to the specified feature
  // group. If some weight has already been added with the same
  // feature, the product of the old and new weights are
  // stored. Returns true iff the weight is added. A weight is not
  // added when the context has ill-formed context involving start-,
  // end-of-sentence marks.
  //
  // For two features to be within the same group, it must satisfy
  // that (1) they have the same future size; (2) the two either have
  // disjoint context or one is the back-off context of the
  // other. Furthermore, for all features in a single group, there
  // must be one and only one other context (not necessarily an active
  // feature) that the feature immediately backs off to (i.e. there is
  // no other context that is the back-off of the first and backs off
  // to the second).
  //
  // Consider for example features with zero look-ahead of the form
  // (input, OUTPUT).
  //
  // - The following two features can be put in the same group because
  // their context is disjoint: (a a a, A A), (b, B B);
  //
  // - The following two features can be put in the same group because
  // one is the back-off context of the other: (a a a, A A), (a a, A
  // A);
  //
  // - The following two features can NOT be put in the same group
  // because there is overlap but neither is the other's back-off: (a
  // a a, A), (a a, A A);
  //
  // - Finally, the following three features cannot be in a same group
  // because the first one can immediately back off to either of the
  // rest: (a a a, A A), (a a, A A), (a a a, A).
  //
  // The easiest way to satisfy the constraints is to create a feature
  // group for each feature template. However, better feature grouping
  // may help improve speed.
  //
  // This may fail if any of input or output labels are non-positive,
  // or if any call to `FeatureGroupBuilder<>::AddWeight()` fails.
  bool AddWeight(size_t group, const std::vector<Label> &input,
                 const std::vector<Label> &output, Weight weight);

  // Returns a newly created `LinearFstData` object or nullptr in case
  // of failure. The caller takes the ownership of the memory. No
  // other methods shall be called after this --- this is enforced by
  // putting the builder at error state, even when a
  // `LinearFstData<>` object is successfully built.
  //
  // This may fail if the call to any `FeatureGroupBuilder<>::Dump()`
  // fails.
  LinearFstData<A> *Dump();

 private:
  bool error_;
  CompactSet<Label, kNoLabel> all_output_labels_;
  std::map<Label, std::set<Label>> word_output_map_, word_feat_map_;
  std::map<Label, std::set<size_t>> feat_groups_;
  std::vector<std::unique_ptr<FeatureGroupBuilder<A>>> groups_;
  size_t max_future_size_;
  Label max_input_label_;
  const SymbolTable *isyms_, *fsyms_, *osyms_;

  LinearFstDataBuilder(const LinearFstDataBuilder &) = delete;
  LinearFstDataBuilder &operator=(const LinearFstDataBuilder &) = delete;
};

// Builds a LinearFstData tailored for a LinearClassifierFst. The
// major difference between an ordinary LinearFstData that works on
// taggers and a LinearFstData that works on classifiers is that
// feature groups are divided into sections by the prediction class
// label. For a prediction label `pred` and a logical group id
// `group`, the actual group id is `group * num_classes + pred -
// 1`.
//
// This layout saves us from recording output labels in each single
// FeatureGroup. Because there is no need for any delaying, stripping
// the output allows features with different shapes but using the same
// set of feature label mapping to reside in a single FeatureGroup.
template <class A>
class LinearClassifierFstDataBuilder {
 public:
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;

  // Constructs a builder for a `num_classes`-class classifier,
  // optinally with associated symbol tables for diagnostic
  // output. The output labels (i.e. prediction) must be in the range
  // of [1, num_classes].
  explicit LinearClassifierFstDataBuilder(size_t num_classes,
                                          const SymbolTable *isyms = nullptr,
                                          const SymbolTable *fsyms = nullptr,
                                          const SymbolTable *osyms = nullptr)
      : error_(false),
        num_classes_(num_classes),
        num_groups_(0),
        builder_(isyms, fsyms, osyms) {}

  // Tests whether the builder has encountered any error. Similar to
  // LinearFstDataBuilder<>::Error().
  bool Error() const { return error_; }

  // Same as LinearFstDataBuilder<>::AddWord().
  bool AddWord(Label word, const std::vector<Label> &features);

  // Adds a logical feature group. Similar to
  // LinearFstDataBuilder<>::AddGroup(), with the exception that the
  // returned group id is the logical group id. Also there is no need
  // for "future" in a classifier.
  int AddGroup();

  // Adds an instance of feature weight to the specified logical
  // feature group. Instead of a vector of output, only a single
  // prediction is needed as the output.
  //
  // This may fail if `pred` is not in the range of [1, num_classes_].
  bool AddWeight(size_t group, const std::vector<Label> &input, Label pred,
                 Weight weight);

  // Returns a newly created `LinearFstData` object or nullptr in case of
  // failure.
  LinearFstData<A> *Dump();

 private:
  std::vector<Label> empty_;
  bool error_;
  size_t num_classes_, num_groups_;
  LinearFstDataBuilder<A> builder_;
};

// Builds a single feature group. Usually used in
// `LinearFstDataBuilder::AddWeight()`. See that method for the
// constraints on grouping features.
template <class A>
class FeatureGroupBuilder {
 public:
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;

  // Constructs a builder with the given future size. All features
  // added to the group will have look-ahead windows of this size.
  FeatureGroupBuilder(size_t future_size, const SymbolTable *fsyms,
                      const SymbolTable *osyms)
      : error_(false), future_size_(future_size), fsyms_(fsyms), osyms_(osyms) {
    // This edge is special; see doc of class `FeatureGroup` on the
    // details.
    start_ = trie_.Insert(trie_.Root(), InputOutputLabel(kNoLabel, kNoLabel));
  }

  // Tests whether the builder has encountered any error. No operation
  // is valid if the builder is already at error state. All other
  // public methods should check this before any actual operations.
  bool Error() const { return error_; }

  // Adds a feature weight with the given context. Returns true iff
  // the weight is added. A weight is not added if it has ill-formed
  // context involving start-, end-of-sentence marks.
  //
  // Note: `input` is the sequence of input
  // features, instead of input labels themselves. `input` must be at
  // least as long as `future_size`; `output` may be empty, but
  // usually should be non-empty because an empty output context is
  // useless in discriminative modelling. All labels in both `input`
  // and `output` must be > 0 (this is checked in
  // `LinearFstDataBuilder::AddWeight()`). See
  // LinearFstDataBuilder<>::AddWeight for more details.
  //
  // This may fail if the input is smaller than the look-ahead window.
  bool AddWeight(const std::vector<Label> &input,
                 const std::vector<Label> &output, Weight weight);

  // Creates an actual FeatureGroup<> object. Connects back-off links;
  // pre-accumulates weights from back-off features. Returns nullptr if
  // there is any violation in unique immediate back-off
  // constraints.
  //
  // Regardless of whether the call succeeds or not, the error flag is
  // always set before this returns, to prevent repeated dumping.
  //
  // TODO(wuke): check overlapping top-level contexts (see
  // `DumpOverlappingContext()` in tests).
  FeatureGroup<A> *Dump(size_t max_future_size);

 private:
  typedef typename FeatureGroup<A>::InputOutputLabel InputOutputLabel;
  typedef typename FeatureGroup<A>::InputOutputLabelHash InputOutputLabelHash;
  typedef typename FeatureGroup<A>::WeightBackLink WeightBackLink;
  // Nested trie topology uses more memory but we can traverse a
  // node's children easily, which is required in `BuildBackLinks()`.
  typedef NestedTrieTopology<InputOutputLabel, InputOutputLabelHash> Topology;
  typedef MutableTrie<InputOutputLabel, WeightBackLink, Topology> Trie;

  // Finds the first node with an arc with `label` following the
  // back-off chain of `parent`. Returns the node index or
  // `kNoTrieNodeId` when not found. The number of hops is stored in
  // `hop` when it is not `nullptr`.
  //
  // This does not fail.
  int FindFirstMatch(InputOutputLabel label, int parent, int *hop) const;

  // Links each node to its immediate back-off. root is linked to -1.
  //
  // This may fail when the unique immediate back-off constraint is
  // violated.
  void BuildBackLinks();

  // Traces back on the back-chain for each node to multiply the
  // weights from back-offs to the node itself.
  //
  // This does not fail.
  void PreAccumulateWeights();

  // Reconstruct the path from trie root to given node for logging.
  bool TrieDfs(const Topology &topology, int cur, int target,
               std::vector<InputOutputLabel> *path) const;
  string TriePath(int node, const Topology &topology) const;

  bool error_;
  size_t future_size_;
  Trie trie_;
  int start_;
  const SymbolTable *fsyms_, *osyms_;

  FeatureGroupBuilder(const FeatureGroupBuilder &) = delete;
  FeatureGroupBuilder &operator=(const FeatureGroupBuilder &) = delete;
};

//
// Implementation of methods in `LinearFstDataBuilder`
//
template <class A>
bool LinearFstDataBuilder<A>::AddWord(Label word,
                                      const std::vector<Label> &features) {
  if (error_) {
    FSTERROR() << "Calling LinearFstDataBuilder<>::AddWord() at error state";
    return false;
  }
  if (word == LinearFstData<A>::kStartOfSentence ||
      word == LinearFstData<A>::kEndOfSentence) {
    LOG(WARNING) << "Ignored: adding boundary label: "
                 << TranslateLabel(word, isyms_)
                 << "(start-of-sentence=" << LinearFstData<A>::kStartOfSentence
                 << ", end-of-sentence=" << LinearFstData<A>::kEndOfSentence
                 << ")";
    return false;
  }
  if (word <= 0) {
    error_ = true;
    FSTERROR() << "Word label must be > 0; got " << word;
    return false;
  }
  if (word > max_input_label_) max_input_label_ = word;
  // Make sure the word hasn't been added before
  if (word_feat_map_.find(word) != word_feat_map_.end()) {
    error_ = true;
    FSTERROR() << "Input word " << TranslateLabel(word, isyms_)
               << " is added twice";
    return false;
  }
  // Store features
  std::set<Label> *feats = &word_feat_map_[word];
  for (size_t i = 0; i < features.size(); ++i) {
    Label feat = features[i];
    if (feat <= 0) {
      error_ = true;
      FSTERROR() << "Feature label must be > 0; got " << feat;
      return false;
    }
    feats->insert(feat);
  }
  return true;
}

template <class A>
bool LinearFstDataBuilder<A>::AddWord(
    Label word, const std::vector<Label> &word_features,
    const std::vector<Label> &possible_output) {
  if (error_) {
    FSTERROR() << "Calling LinearFstDataBuilder<>::AddWord() at error state";
    return false;
  }
  if (!AddWord(word, word_features)) return false;
  // Store possible output constraint
  if (possible_output.empty()) {
    error_ = true;
    FSTERROR() << "Empty possible output constraint; "
               << "use the two-parameter version if no constraint is need.";
    return false;
  }
  std::set<Label> *outputs = &word_output_map_[word];
  for (size_t i = 0; i < possible_output.size(); ++i) {
    Label output = possible_output[i];
    if (output == LinearFstData<A>::kStartOfSentence ||
        output == LinearFstData<A>::kEndOfSentence) {
      LOG(WARNING) << "Ignored: word = " << TranslateLabel(word, isyms_)
                   << ": adding boundary label as possible output: " << output
                   << "(start-of-sentence="
                   << LinearFstData<A>::kStartOfSentence
                   << ", end-of-sentence=" << LinearFstData<A>::kEndOfSentence
                   << ")";
      continue;
    }
    if (output <= 0) {
      error_ = true;
      FSTERROR() << "Output label must be > 0; got " << output;
      return false;
    }
    outputs->insert(output);
    all_output_labels_.Insert(output);
  }
  return true;
}

template <class A>
inline int LinearFstDataBuilder<A>::AddGroup(size_t future_size) {
  if (error_) {
    FSTERROR() << "Calling LinearFstDataBuilder<>::AddGroup() at error state";
    return -1;
  }
  size_t ret = groups_.size();
  groups_.emplace_back(new FeatureGroupBuilder<A>(future_size, fsyms_, osyms_));
  if (future_size > max_future_size_) max_future_size_ = future_size;
  return ret;
}

template <class A>
bool LinearFstDataBuilder<A>::AddWeight(size_t group,
                                        const std::vector<Label> &input,
                                        const std::vector<Label> &output,
                                        Weight weight) {
  if (error_) {
    FSTERROR() << "Calling LinearFstDataBuilder<>::AddWeight() at error state";
    return false;
  }
  // Check well-formedness of boundary marks on the input.
  {
    bool start_in_middle = false, end_in_middle = false;
    for (int i = 1; i < input.size(); ++i) {
      if (input[i] == LinearFstData<A>::kStartOfSentence &&
          input[i - 1] != LinearFstData<A>::kStartOfSentence)
        start_in_middle = true;
      if (input[i - 1] == LinearFstData<A>::kEndOfSentence &&
          input[i] != LinearFstData<A>::kEndOfSentence)
        end_in_middle = true;
    }
    if (start_in_middle) {
      LOG(WARNING) << "Ignored: start-of-sentence in the middle of the input!";
      LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
      LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
      return false;
    }
    if (end_in_middle) {
      LOG(WARNING) << "Ignored: end-of-sentence in the middle of the input!";
      LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
      LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
      return false;
    }
  }
  // Check well-formedness of boundary marks on the output.
  {
    bool non_first_start = false, non_last_end = false;
    for (int i = 1; i < output.size(); ++i) {
      if (output[i] == LinearFstData<A>::kStartOfSentence)
        non_first_start = true;
      if (output[i - 1] == LinearFstData<A>::kEndOfSentence)
        non_last_end = true;
    }
    if (non_first_start) {
      LOG(WARNING) << "Ignored: start-of-sentence not appearing "
                   << "as the first label in the output!";
      LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
      LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
      return false;
    }
    if (non_last_end) {
      LOG(WARNING) << "Ignored: end-of-sentence not appearing "
                   << "as the last label in the output!";
      LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
      LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
      return false;
    }
  }

  for (size_t i = 0; i < input.size(); ++i) {
    Label feat = input[i];
    if (feat != LinearFstData<A>::kStartOfSentence &&
        feat != LinearFstData<A>::kEndOfSentence && feat <= 0) {
      error_ = true;
      FSTERROR() << "Feature label must be > 0; got " << feat;
      return false;
    }
    feat_groups_[feat].insert(group);
  }
  for (size_t i = 0; i < output.size(); ++i) {
    Label label = output[i];
    if (label != LinearFstData<A>::kStartOfSentence &&
        label != LinearFstData<A>::kEndOfSentence && label <= 0) {
      error_ = true;
      FSTERROR() << "Output label must be > 0; got " << label;
      return false;
    }
    if (label != LinearFstData<A>::kStartOfSentence &&
        label != LinearFstData<A>::kEndOfSentence)
      all_output_labels_.Insert(label);
  }

  // Everything looks good at this point (more checks on the way in
  // the feature group). Add this feature weight.
  bool added = groups_[group]->AddWeight(input, output, weight);
  if (groups_[group]->Error()) {
    error_ = true;
    FSTERROR() << "FeatureGroupBuilder<>::AddWeight() failed";
    return false;
  }
  return added;
}

template <class A>
LinearFstData<A> *LinearFstDataBuilder<A>::Dump() {
  if (error_) {
    FSTERROR() << "Calling LinearFstDataBuilder<>::Dump() at error state";
    return nullptr;
  }

  std::unique_ptr<LinearFstData<A>> data(new LinearFstData<A>());
  data->max_future_size_ = max_future_size_;
  data->max_input_label_ = max_input_label_;

  // Feature groups; free builders after it's dumped.
  data->groups_.resize(groups_.size());
  for (int group = 0; group != groups_.size(); ++group) {
    FeatureGroup<A> *new_group = groups_[group]->Dump(max_future_size_);
    if (new_group == nullptr) {
      error_ = true;
      FSTERROR() << "Error in dumping group " << group;
      return nullptr;
    }
    data->groups_[group].reset(new_group);
    groups_[group].reset();
    VLOG(1) << "Group " << group << ": " << new_group->Stats();
  }

  // Per-group feature mapping
  data->group_feat_map_.Init(data->NumGroups(), max_input_label_ + 1);
  for (Label word = 1; word <= max_input_label_; ++word) {
    typename std::map<Label, std::set<Label>>::const_iterator it =
        word_feat_map_.find(word);
    if (it == word_feat_map_.end()) continue;
    for (typename std::set<Label>::const_iterator oit = it->second.begin();
         oit != it->second.end(); ++oit) {
      Label feat = *oit;
      typename std::map<Label, std::set<size_t>>::const_iterator jt =
          feat_groups_.find(feat);
      if (jt == feat_groups_.end()) continue;
      for (std::set<size_t>::const_iterator git = jt->second.begin();
           git != jt->second.end(); ++git) {
        size_t group_id = *git;
        if (!data->group_feat_map_.Set(group_id, word, feat)) {
          error_ = true;
          return nullptr;
        }
      }
    }
  }

  // Possible output labels
  {
    std::vector<typename LinearFstData<A>::InputAttribute> *input_attribs =
        &data->input_attribs_;
    std::vector<Label> *output_pool = &data->output_pool_;
    input_attribs->resize(max_input_label_ + 1);
    for (Label word = 0; word <= max_input_label_; ++word) {
      typename std::map<Label, std::set<Label>>::const_iterator it =
          word_output_map_.find(word);
      if (it == word_output_map_.end()) {
        (*input_attribs)[word].output_begin = 0;
        (*input_attribs)[word].output_length = 0;
      } else {
        (*input_attribs)[word].output_begin = output_pool->size();
        (*input_attribs)[word].output_length = it->second.size();
        for (typename std::set<Label>::const_iterator oit = it->second.begin();
             oit != it->second.end(); ++oit) {
          Label olabel = *oit;
          output_pool->push_back(olabel);
        }
      }
    }
  }

  for (typename CompactSet<Label, kNoLabel>::const_iterator it =
           all_output_labels_.Begin();
       it != all_output_labels_.End(); ++it)
    data->output_set_.push_back(*it);

  error_ = true;  // prevent future calls on this object
  return data.release();
}

//
// Implementation of methods in `LinearClassifierFstDataBuilder`
//
template <class A>
inline bool LinearClassifierFstDataBuilder<A>::AddWord(
    Label word, const std::vector<Label> &features) {
  if (error_) {
    FSTERROR() << "Calling LinearClassifierFstDataBuilder<>::AddWord() at "
                  "error state";
    return false;
  }
  bool added = builder_.AddWord(word, features);
  if (builder_.Error()) error_ = true;
  return added;
}

template <class A>
inline int LinearClassifierFstDataBuilder<A>::AddGroup() {
  if (error_) {
    FSTERROR() << "Calling LinearClassifierFstDataBuilder<>::AddGroup() at "
                  "error state";
    return -1;
  }
  for (int i = 0; i < num_classes_; ++i) builder_.AddGroup(0);
  if (builder_.Error()) {
    error_ = true;
    return -1;
  }
  return num_groups_++;
}

template <class A>
inline bool LinearClassifierFstDataBuilder<A>::AddWeight(
    size_t group, const std::vector<Label> &input, Label pred, Weight weight) {
  if (error_) {
    FSTERROR() << "Calling LinearClassifierFstDataBuilder<>::AddWeight() at "
                  "error state";
    return false;
  }
  if (pred <= 0 || pred > num_classes_) {
    FSTERROR() << "Out-of-range prediction label: " << pred
               << " (num classes = " << num_classes_ << ")";
    error_ = true;
    return false;
  }
  size_t real_group = group * num_classes_ + pred - 1;
  bool added = builder_.AddWeight(real_group, input, empty_, weight);
  if (builder_.Error()) error_ = true;
  return added;
}

template <class A>
inline LinearFstData<A> *LinearClassifierFstDataBuilder<A>::Dump() {
  if (error_) {
    FSTERROR()
        << "Calling LinearClassifierFstDataBuilder<>::Dump() at error state";
    return nullptr;
  }
  LinearFstData<A> *data = builder_.Dump();
  error_ = true;
  return data;
}

//
// Implementation of methods in `FeatureGroupBuilder`
//
template <class A>
bool FeatureGroupBuilder<A>::AddWeight(const std::vector<Label> &input,
                                       const std::vector<Label> &output,
                                       Weight weight) {
  if (error_) {
    FSTERROR() << "Calling FeatureGroupBuilder<>::AddWeight() at error state";
    return false;
  }

  // `LinearFstDataBuilder<>::AddWeight()` ensures prefix/suffix
  // properties for us. We can directly count.
  int num_input_start = 0;
  while (num_input_start < input.size() &&
         input[num_input_start] == LinearFstData<A>::kStartOfSentence)
    ++num_input_start;
  int num_output_start = 0;
  while (num_output_start < output.size() &&
         output[num_output_start] == LinearFstData<A>::kStartOfSentence)
    ++num_output_start;
  int num_input_end = 0;
  for (int i = input.size() - 1;
       i >= 0 && input[i] == LinearFstData<A>::kEndOfSentence; --i)
    ++num_input_end;
  int num_output_end = 0;
  for (int i = output.size() - 1;
       i >= 0 && output[i] == LinearFstData<A>::kEndOfSentence; --i)
    ++num_output_end;

  DCHECK_LE(num_output_end, 1);

  if (input.size() - num_input_start < future_size_) {
    LOG(WARNING) << "Ignored: start-of-sentence in the future!";
    LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
    LOG(WARNING) << "\tOutput: " << JoinLabels(output, fsyms_);
    return false;
  }
  if (num_input_start > 0 &&
      input.size() - future_size_ - num_input_start <
          output.size() - num_output_start) {
    LOG(WARNING) << "Ignored: matching start-of-sentence with actual output!";
    LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
    LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
    return false;
  }
  if (num_output_start > 0 &&
      input.size() - future_size_ - num_input_start >
          output.size() - num_output_start) {
    LOG(WARNING) << "Ignored: matching start-of-sentence with actual input!";
    LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
    LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
    return false;
  }
  // The following two require `num_output_end` <= 1.
  if (num_input_end > future_size_ && num_input_end - future_size_ != 1) {
    LOG(WARNING) << "Ignored: matching end-of-sentence with actual output!";
    LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
    LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
    return false;
  }
  if (num_output_end > 0 &&
      ((input.size() == future_size_ && future_size_ != num_input_end) ||
       (input.size() > future_size_ &&
        num_input_end != future_size_ + num_output_end))) {
    LOG(WARNING) << "Ignored: matching end-of-sentence with actual input!";
    LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
    LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
    return false;
  }
  // Check if the context has no other labels than boundary marks
  // (such features are useless).
  if (num_input_start + num_input_end == input.size() &&
      num_output_start + num_output_end == output.size()) {
    LOG(WARNING)
        << "Ignored: feature context consisting of only boundary marks!";
    LOG(WARNING) << "\tInput: " << JoinLabels(input, fsyms_);
    LOG(WARNING) << "\tOutput: " << JoinLabels(output, osyms_);
    return false;
  }

  // Start point for insertion in the trie. Insert at `start_` iff the
  // beginning of the context is non-consumed start-of-sentence.
  int cur = (num_input_start == 0 && num_output_start <= future_size_)
                ? trie_.Root()
                : start_;
  // Skip all input start-of-sentence marks
  size_t ipos = num_input_start;
  // Skip to keep at most `future_size_` start-of-sentence marks
  size_t opos =
      num_output_start <= future_size_ ? 0 : num_output_start - future_size_;
  // Skip `num_output_end` end-of-sentence marks on both input and output
  size_t iend = !input.empty() ? input.size() - num_output_end : 0,
         oend = output.size() - num_output_end;
  // Further, when output is empty, keep at most `future_size_`
  // end-of-sentence marks on input.
  if (output.empty() && num_input_end > future_size_)
    iend = input.size() - num_input_end + future_size_;

  // Actual feature context is (input[ipos:iend], output[opos:oend]).

  // Pad `kNoLabel` as don't cares on the shorter of actual `input`
  // and `output`.
  const size_t effective_input_size = iend - ipos,
               effective_output_size = oend - opos;
  if (effective_input_size > effective_output_size) {
    for (size_t pad = effective_input_size - effective_output_size; pad != 0;
         --pad, ++ipos)
      cur = trie_.Insert(cur, InputOutputLabel(input[ipos], kNoLabel));
  } else if (effective_input_size < effective_output_size) {
    for (size_t pad = effective_output_size - effective_input_size; pad != 0;
         --pad, ++opos)
      cur = trie_.Insert(cur, InputOutputLabel(kNoLabel, output[opos]));
  }
  CHECK_EQ(iend - ipos, oend - opos);
  for (; ipos != iend; ++ipos, ++opos)
    cur = trie_.Insert(cur, InputOutputLabel(input[ipos], output[opos]));
  // We only need to attach final weight when there is an output
  // end-of-sentence. When there is only end-of-sentence on the input,
  // they are all consumed as the end-of-sentence paddings from
  // `LinearFstImpl<>::ShiftBuffer()`. `LinearFstImpl<>::Expand()`
  // and `LinearFstImpl<>::MatchInput()` ensures no other
  // transition takes place after consuming the padding.
  if (num_output_end > 0 || (output.empty() && num_input_end > future_size_))
    trie_[cur].final_weight = Times(trie_[cur].final_weight, weight);
  else
    trie_[cur].weight = Times(trie_[cur].weight, weight);

  return true;
}

template <class A>
FeatureGroup<A> *FeatureGroupBuilder<A>::Dump(size_t max_future_size) {
  if (error_) {
    FSTERROR() << "Calling FeatureGroupBuilder<>::PreAccumulateWeights() "
               << "at error state";
    return nullptr;
  }

  if (max_future_size < future_size_) {
    error_ = true;
    FSTERROR() << "max_future_size (= " << max_future_size
               << ") is smaller the builder's future_size (= " << future_size_
               << ")";
    return nullptr;
  }

  BuildBackLinks();
  if (error_) return nullptr;
  PreAccumulateWeights();  // does not fail

  FeatureGroup<A> *ret =
      new FeatureGroup<A>(max_future_size - future_size_, start_);

  // Walk around the trie to compute next states
  ret->next_state_.resize(trie_.NumNodes());
  const Topology &topology = trie_.TrieTopology();
  for (int i = 0; i < topology.NumNodes(); ++i) {
    int next = i;
    while (next != topology.Root() && topology.ChildrenOf(next).empty() &&
           trie_[next].final_weight ==
               trie_[trie_[next].back_link].final_weight)
      next = trie_[next].back_link;
    ret->next_state_[i] = next;
  }

  // Copy the trie
  typename FeatureGroup<A>::Trie store_trie(trie_);
  ret->trie_.swap(store_trie);

  // Put the builder at error state to prevent repeated call of `Dump()`.
  error_ = true;
  return ret;
}

template <class A>
int FeatureGroupBuilder<A>::FindFirstMatch(InputOutputLabel label, int parent,
                                           int *hop) const {
  int hop_count = 0;
  int ret = kNoTrieNodeId;
  for (; parent >= 0; parent = trie_[parent].back_link, ++hop_count) {
    int next = trie_.Find(parent, label);
    if (next != kNoTrieNodeId) {
      ret = next;
      break;
    }
  }
  if (hop != nullptr) *hop = hop_count;
  return ret;
}

template <class A>
void FeatureGroupBuilder<A>::BuildBackLinks() {
  // Breadth first search from the root. In the case where we only
  // have the input label, the immedate back-off is simply the longest
  // suffix of the current node that is also in the trie. For a node
  // reached from its parent with label L, we can simply walk through
  // the parent's back-off chain to find the first state with an arc
  // of the same label L. The uniqueness is always
  // guanranteed. However, in the case with both input and output
  // labels, it is possible to back off by removing first labels from
  // either side, which in general causes non-uniqueness.

  const Topology &topology = trie_.TrieTopology();
  std::queue<int> q;  // all enqueued or visited nodes have known links

  // Note: nodes have back link initialized to -1 in their
  // constructor.
  q.push(trie_.Root());
  while (!error_ && !q.empty()) {
    int parent = q.front();
    q.pop();
    // Find links for every child
    const typename Topology::NextMap &children = topology.ChildrenOf(parent);
    for (typename Topology::NextMap::const_iterator eit = children.begin();
         eit != children.end(); ++eit) {
      const std::pair<InputOutputLabel, int> &edge = *eit;
      InputOutputLabel label = edge.first;
      int child = edge.second;
      if (label.input == kNoLabel || label.output == kNoLabel) {
        // Label pairs from root to here all have one and only one
        // `kNoLabel` on the same side; equivalent to the
        // "longest-suffix" case.
        trie_[child].back_link =
            FindFirstMatch(label, trie_[parent].back_link, nullptr);
      } else {
        // Neither side is `kNoLabel` at this point, there are
        // three possible ways to back-off: if the parent backs
        // off to some context with only one side non-empty, the
        // empty side may remain empty; or else an exact match of
        // both sides is needed. Try to find all three possible
        // backs and look for the closest one (in terms of hops
        // along the parent's back-off chain).
        int only_input_hop, only_output_hop, full_hop;
        int only_input_link =
                FindFirstMatch(InputOutputLabel(label.input, kNoLabel), parent,
                               &only_input_hop),
            only_output_link =
                FindFirstMatch(InputOutputLabel(kNoLabel, label.output), parent,
                               &only_output_hop),
            full_link =
                FindFirstMatch(label, trie_[parent].back_link, &full_hop);
        if (only_input_link != -1 && only_output_link != -1) {
          error_ = true;
          FSTERROR() << "Branching back-off chain:\n"
                     << "\tnode " << child << ": " << TriePath(child, topology)
                     << "\n"
                     << "\tcan back-off to node " << only_input_link << ": "
                     << TriePath(only_input_link, topology) << "\n"
                     << "\tcan back-off to node " << only_output_link << ": "
                     << TriePath(only_output_link, topology);
          return;
        } else if (full_link != -1) {
          ++full_hop;
          if (full_hop <= only_input_hop && full_hop <= only_output_hop) {
            trie_[child].back_link = full_link;
          } else {
            error_ = true;
            int problem_link = only_input_link != kNoTrieNodeId
                                   ? only_input_link
                                   : only_output_link;
            CHECK_NE(problem_link, kNoTrieNodeId);
            FSTERROR() << "Branching back-off chain:\n"
                       << "\tnode " << child << ": "
                       << TriePath(child, topology) << "\n"
                       << "\tcan back-off to node " << full_link << ": "
                       << TriePath(full_link, topology) << "\n"
                       << "tcan back-off to node " << problem_link << ": "
                       << TriePath(problem_link, topology);
            return;
          }
        } else {
          trie_[child].back_link =
              only_input_link != -1 ? only_input_link : only_output_link;
        }
      }
      if (error_) break;
      // Point to empty context (root) when no back-off can be found
      if (trie_[child].back_link == -1) trie_[child].back_link = 0;
      q.push(child);
    }
  }
}

template <class A>
void FeatureGroupBuilder<A>::PreAccumulateWeights() {
  std::vector<bool> visited(trie_.NumNodes(), false);
  visited[trie_.Root()] = true;

  for (size_t i = 0; i != trie_.NumNodes(); ++i) {
    std::stack<int> back_offs;
    for (int j = i; !visited[j]; j = trie_[j].back_link) back_offs.push(j);
    while (!back_offs.empty()) {
      int j = back_offs.top();
      back_offs.pop();
      WeightBackLink &node = trie_[j];
      node.weight = Times(node.weight, trie_[node.back_link].weight);
      node.final_weight =
          Times(node.final_weight, trie_[node.back_link].final_weight);
      visited[j] = true;
    }
  }
}

template <class A>
bool FeatureGroupBuilder<A>::TrieDfs(
    const Topology &topology, int cur, int target,
    std::vector<InputOutputLabel> *path) const {
  if (cur == target) return true;
  const typename Topology::NextMap &children = topology.ChildrenOf(cur);
  for (typename Topology::NextMap::const_iterator eit = children.begin();
       eit != children.end(); ++eit) {
    const std::pair<InputOutputLabel, int> &edge = *eit;
    path->push_back(edge.first);
    if (TrieDfs(topology, edge.second, target, path)) return true;
    path->pop_back();
  }
  return false;
}

template <class A>
string FeatureGroupBuilder<A>::TriePath(int node,
                                        const Topology &topology) const {
  std::vector<InputOutputLabel> labels;
  TrieDfs(topology, topology.Root(), node, &labels);
  bool first = true;
  std::ostringstream strm;
  for (typename std::vector<InputOutputLabel>::const_iterator it =
           labels.begin();
       it != labels.end(); ++it) {
    InputOutputLabel i = *it;
    if (first)
      first = false;
    else
      strm << ", ";
    strm << "(" << TranslateLabel(i.input, fsyms_) << ", "
         << TranslateLabel(i.output, osyms_) << ")";
  }
  return strm.str();
}

inline string TranslateLabel(int64_t label, const SymbolTable *syms) {
  string ret;
  if (syms != nullptr) ret += syms->Find(label);
  if (ret.empty()) {
    std::ostringstream strm;
    strm << '<' << label << '>';
    ret = strm.str();
  }
  return ret;
}

template <class Iterator>
string JoinLabels(Iterator begin, Iterator end, const SymbolTable *syms) {
  if (begin == end) return "<empty>";
  std::ostringstream strm;
  bool first = true;
  for (Iterator it = begin; it != end; ++it) {
    if (first)
      first = false;
    else
      strm << '|';
    strm << TranslateLabel(*it, syms);
  }
  return strm.str();
}

template <class Label>
string JoinLabels(const std::vector<Label> &labels, const SymbolTable *syms) {
  return JoinLabels(labels.begin(), labels.end(), syms);
}

template <class A>
typename A::Label GuessStartOrEnd(std::vector<typename A::Label> *sequence,
                                  typename A::Label boundary) {
  const size_t length = sequence->size();
  std::vector<bool> non_boundary_on_left(length, false),
      non_boundary_on_right(length, false);
  for (size_t i = 1; i < length; ++i) {
    non_boundary_on_left[i] =
        non_boundary_on_left[i - 1] || (*sequence)[i - 1] != boundary;
    non_boundary_on_right[length - 1 - i] = non_boundary_on_right[length - i] ||
                                            (*sequence)[length - i] != boundary;
  }
  int unresolved = 0;
  for (size_t i = 0; i < length; ++i) {
    if ((*sequence)[i] != boundary) continue;
    const bool left = non_boundary_on_left[i], right = non_boundary_on_right[i];
    if (left && right) {
      // Boundary in the middle
      LOG(WARNING) << "Boundary label in the middle of the sequence! position: "
                   << i << "; boundary: " << boundary
                   << "; sequence: " << JoinLabels(*sequence, nullptr);
      LOG(WARNING)
          << "This is an invalid sequence anyway so I will set it to start.";
      (*sequence)[i] = LinearFstData<A>::kStartOfSentence;
    } else if (left && !right) {
      // Can only be end
      (*sequence)[i] = LinearFstData<A>::kEndOfSentence;
    } else if (!left && right) {
      // Can only be start
      (*sequence)[i] = LinearFstData<A>::kStartOfSentence;
    } else {
      // !left && !right; can't really tell
      ++unresolved;
    }
  }
  return unresolved;
}

}  // namespace fst

#endif  // FST_EXTENSIONS_LINEAR_LINEAR_FST_DATA_BUILDER_H_
