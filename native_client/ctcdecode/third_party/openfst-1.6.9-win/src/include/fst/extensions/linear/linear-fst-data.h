// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Data structures for storing and looking up the actual feature weights.

#ifndef FST_EXTENSIONS_LINEAR_LINEAR_FST_DATA_H_
#define FST_EXTENSIONS_LINEAR_LINEAR_FST_DATA_H_

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <fst/compat.h>
#include <fst/fst.h>

#include <fst/extensions/linear/trie.h>

namespace fst {

// Forward declarations
template <class A>
class LinearFstDataBuilder;
template <class A>
class FeatureGroup;

// Immutable data storage of the feature weights in a linear
// model. Produces state tuples that represent internal states of a
// LinearTaggerFst. Object of this class can only be constructed via
// either `LinearFstDataBuilder::Dump()` or `LinearFstData::Read()`
// and usually used as refcount'd object shared across mutiple
// `LinearTaggerFst` copies.
//
// TODO(wuke): more efficient trie implementation
template <class A>
class LinearFstData {
 public:
  friend class LinearFstDataBuilder<A>;  // For builder access

  typedef typename A::Label Label;
  typedef typename A::Weight Weight;

  // Sentence boundary labels. Both of them are negative labels other
  // than `kNoLabel`.
  static const Label kStartOfSentence;
  static const Label kEndOfSentence;

  // Constructs empty data; for non-trivial ways of construction see
  // `Read()` and `LinearFstDataBuilder`.
  LinearFstData()
      : max_future_size_(0), max_input_label_(1), input_attribs_(1) {}

  // Appends the state tuple of the start state to `output`, where
  // each tuple holds the node ids of a trie for each feature group.
  void EncodeStartState(std::vector<Label> *output) const {
    for (int i = 0; i < NumGroups(); ++i) output->push_back(GroupStartState(i));
  }

  // Takes a transition from the trie states stored in
  // `(trie_state_begin, trie_state_end)` with input label `ilabel`
  // and output label `olabel`; appends the destination state tuple to
  // `next` and multiplies the weight of the transition onto
  // `weight`. `next` should be the shifted input buffer of the caller
  // in `LinearTaggerFstImpl` (i.e. of size `LinearTaggerFstImpl::delay_`;
  // the last element is `ilabel`).
  template <class Iterator>
  void TakeTransition(Iterator buffer_end, Iterator trie_state_begin,
                      Iterator trie_state_end, Label ilabel, Label olabel,
                      std::vector<Label> *next, Weight *weight) const;

  // Returns the final weight of the given trie state sequence.
  template <class Iterator>
  Weight FinalWeight(Iterator trie_state_begin, Iterator trie_state_end) const;

  // Returns the start trie state of the given group.
  Label GroupStartState(int group_id) const {
    return groups_[group_id]->Start();
  }

  // Takes a transition only within the given group. Returns the
  // destination trie state and multiplies the weight onto `weight`.
  Label GroupTransition(int group_id, int trie_state, Label ilabel,
                        Label olabel, Weight *weight) const;

  // Returns the final weight of the given trie state in the given group.
  Weight GroupFinalWeight(int group_id, int trie_state) const {
    return groups_[group_id]->FinalWeight(trie_state);
  }

  Label MinInputLabel() const { return 1; }

  Label MaxInputLabel() const { return max_input_label_; }

  // Returns the maximum future size of all feature groups. Future is
  // the look-ahead window of a feature, e.g. if a feature looks at
  // the next 2 words after the current input, then the future size is
  // 2. There is no look-ahead for output. Features inside a single
  // `FeatureGroup` must have equal future size.
  size_t MaxFutureSize() const { return max_future_size_; }

  // Returns the number of feature groups
  size_t NumGroups() const { return groups_.size(); }

  // Returns the range of possible output labels for an input label.
  std::pair<typename std::vector<Label>::const_iterator,
            typename std::vector<Label>::const_iterator>
  PossibleOutputLabels(Label word) const;

  static LinearFstData<A> *Read(std::istream &strm);  // NOLINT
  std::ostream &Write(std::ostream &strm) const;      // NOLINT

 private:
  // Offsets in `output_pool_`
  struct InputAttribute {
    size_t output_begin, output_length;

    std::istream &Read(std::istream &strm);         // NOLINT
    std::ostream &Write(std::ostream &strm) const;  // NOLINT
  };

  // Mapping from input label to per-group feature label
  class GroupFeatureMap;

  // Translates the input label into input feature label of group
  // `group`; returns `kNoLabel` when there is no feature for that
  // group.
  Label FindFeature(size_t group, Label word) const;

  size_t max_future_size_;
  Label max_input_label_;
  std::vector<std::unique_ptr<const FeatureGroup<A>>> groups_;
  std::vector<InputAttribute> input_attribs_;
  std::vector<Label> output_pool_, output_set_;
  GroupFeatureMap group_feat_map_;

  LinearFstData(const LinearFstData &) = delete;
  LinearFstData &operator=(const LinearFstData &) = delete;
};

template <class A>
const typename A::Label LinearFstData<A>::kStartOfSentence = -3;
template <class A>
const typename A::Label LinearFstData<A>::kEndOfSentence = -2;

template <class A>
template <class Iterator>
void LinearFstData<A>::TakeTransition(Iterator buffer_end,
                                      Iterator trie_state_begin,
                                      Iterator trie_state_end, Label ilabel,
                                      Label olabel, std::vector<Label> *next,
                                      Weight *weight) const {
  DCHECK_EQ(trie_state_end - trie_state_begin, groups_.size());
  DCHECK(ilabel > 0 || ilabel == kEndOfSentence);
  DCHECK(olabel > 0 || olabel == kStartOfSentence);
  size_t group_id = 0;
  for (Iterator it = trie_state_begin; it != trie_state_end; ++it, ++group_id) {
    size_t delay = groups_[group_id]->Delay();
    // On the buffer, there may also be `kStartOfSentence` from the
    // initial empty buffer.
    Label real_ilabel = delay == 0 ? ilabel : *(buffer_end - delay);
    next->push_back(
        GroupTransition(group_id, *it, real_ilabel, olabel, weight));
  }
}

template <class A>
typename A::Label LinearFstData<A>::GroupTransition(int group_id,
                                                    int trie_state,
                                                    Label ilabel, Label olabel,
                                                    Weight *weight) const {
  Label group_ilabel = FindFeature(group_id, ilabel);
  return groups_[group_id]->Walk(trie_state, group_ilabel, olabel, weight);
}

template <class A>
template <class Iterator>
inline typename A::Weight LinearFstData<A>::FinalWeight(
    Iterator trie_state_begin, Iterator trie_state_end) const {
  DCHECK_EQ(trie_state_end - trie_state_begin, groups_.size());
  size_t group_id = 0;
  Weight accum = Weight::One();
  for (Iterator it = trie_state_begin; it != trie_state_end; ++it, ++group_id)
    accum = Times(accum, GroupFinalWeight(group_id, *it));
  return accum;
}

template <class A>
inline std::pair<typename std::vector<typename A::Label>::const_iterator,
                 typename std::vector<typename A::Label>::const_iterator>
LinearFstData<A>::PossibleOutputLabels(Label word) const {
  const InputAttribute &attrib = input_attribs_[word];
  if (attrib.output_length == 0)
    return std::make_pair(output_set_.begin(), output_set_.end());
  else
    return std::make_pair(
        output_pool_.begin() + attrib.output_begin,
        output_pool_.begin() + attrib.output_begin + attrib.output_length);
}

template <class A>
inline LinearFstData<A> *LinearFstData<A>::Read(std::istream &strm) {  // NOLINT
  std::unique_ptr<LinearFstData<A>> data(new LinearFstData<A>());
  ReadType(strm, &(data->max_future_size_));
  ReadType(strm, &(data->max_input_label_));
  // Feature groups
  size_t num_groups = 0;
  ReadType(strm, &num_groups);
  data->groups_.resize(num_groups);
  for (size_t i = 0; i < num_groups; ++i)
    data->groups_[i].reset(FeatureGroup<A>::Read(strm));
  // Other data
  ReadType(strm, &(data->input_attribs_));
  ReadType(strm, &(data->output_pool_));
  ReadType(strm, &(data->output_set_));
  ReadType(strm, &(data->group_feat_map_));
  if (strm) {
    return data.release();
  } else {
    return nullptr;
  }
}

template <class A>
inline std::ostream &LinearFstData<A>::Write(
    std::ostream &strm) const {  // NOLINT
  WriteType(strm, max_future_size_);
  WriteType(strm, max_input_label_);
  // Feature groups
  WriteType(strm, groups_.size());
  for (size_t i = 0; i < groups_.size(); ++i) {
    groups_[i]->Write(strm);
  }
  // Other data
  WriteType(strm, input_attribs_);
  WriteType(strm, output_pool_);
  WriteType(strm, output_set_);
  WriteType(strm, group_feat_map_);
  return strm;
}

template <class A>
typename A::Label LinearFstData<A>::FindFeature(size_t group,
                                                Label word) const {
  DCHECK(word > 0 || word == kStartOfSentence || word == kEndOfSentence);
  if (word == kStartOfSentence || word == kEndOfSentence)
    return word;
  else
    return group_feat_map_.Find(group, word);
}

template <class A>
inline std::istream &LinearFstData<A>::InputAttribute::Read(
    std::istream &strm) {  // NOLINT
  ReadType(strm, &output_begin);
  ReadType(strm, &output_length);
  return strm;
}

template <class A>
inline std::ostream &LinearFstData<A>::InputAttribute::Write(
    std::ostream &strm) const {  // NOLINT
  WriteType(strm, output_begin);
  WriteType(strm, output_length);
  return strm;
}

// Forward declaration
template <class A>
class FeatureGroupBuilder;

// An immutable grouping of features with similar context shape. Like
// `LinearFstData`, this can only be constructed via `Read()` or
// via its builder.
//
// Internally it uses a trie to store all feature n-grams and their
// weights. The label of a trie edge is a pair (feat, olabel) of
// labels. They can be either positive (ordinary label), `kNoLabel`,
// `kStartOfSentence`, or `kEndOfSentence`. `kNoLabel` usually means
// matching anything, with one exception: from the root of the trie,
// there is a special (kNoLabel, kNoLabel) that leads to the implicit
// start-of-sentence state. This edge is never actually matched
// (`FindFirstMatch()` ensures this).
template <class A>
class FeatureGroup {
 public:
  friend class FeatureGroupBuilder<A>;  // for builder access

  typedef typename A::Label Label;
  typedef typename A::Weight Weight;

  int Start() const { return start_; }

  // Finds destination node from `cur` by consuming `ilabel` and
  // `olabel`. The transition weight is multiplied onto `weight`.
  int Walk(int cur, Label ilabel, Label olabel, Weight *weight) const;

  // Returns the final weight of the current trie state. Only valid if
  // the state is already known to be part of a final state (see
  // `LinearFstData<>::CanBeFinal()`).
  Weight FinalWeight(int trie_state) const {
    return trie_[trie_state].final_weight;
  }

  static FeatureGroup<A> *Read(std::istream &strm) {  // NOLINT
    size_t delay;
    ReadType(strm, &delay);
    int start;
    ReadType(strm, &start);
    Trie trie;
    ReadType(strm, &trie);
    std::unique_ptr<FeatureGroup<A>> ret(new FeatureGroup<A>(delay, start));
    ret->trie_.swap(trie);
    ReadType(strm, &ret->next_state_);
    if (strm) {
      return ret.release();
    } else {
      return nullptr;
    }
  }

  std::ostream &Write(std::ostream &strm) const {  // NOLINT
    WriteType(strm, delay_);
    WriteType(strm, start_);
    WriteType(strm, trie_);
    WriteType(strm, next_state_);
    return strm;
  }

  size_t Delay() const { return delay_; }

  string Stats() const;

 private:
  // Label along the arcs on the trie. `kNoLabel` means anything
  // (non-negative label) can match; both sides holding `kNoLabel`
  // is not allow; otherwise the label is > 0 (enforced by
  // `LinearFstDataBuilder::AddWeight()`).
  struct InputOutputLabel;
  struct InputOutputLabelHash;

  // Data to be stored on the trie
  struct WeightBackLink {
    int back_link;
    Weight weight, final_weight;

    WeightBackLink()
        : back_link(kNoTrieNodeId),
          weight(Weight::One()),
          final_weight(Weight::One()) {}

    std::istream &Read(std::istream &strm) {  // NOLINT
      ReadType(strm, &back_link);
      ReadType(strm, &weight);
      ReadType(strm, &final_weight);
      return strm;
    }

    std::ostream &Write(std::ostream &strm) const {  // NOLINT
      WriteType(strm, back_link);
      WriteType(strm, weight);
      WriteType(strm, final_weight);
      return strm;
    }
  };

  typedef FlatTrieTopology<InputOutputLabel, InputOutputLabelHash> Topology;
  typedef MutableTrie<InputOutputLabel, WeightBackLink, Topology> Trie;

  explicit FeatureGroup(size_t delay, int start)
      : delay_(delay), start_(start) {}

  // Finds the first node with an arc with `label` following the
  // back-off chain of `parent`. Returns the node index or
  // `kNoTrieNodeId` when not found.
  int FindFirstMatch(InputOutputLabel label, int parent) const;

  size_t delay_;
  int start_;
  Trie trie_;
  // Where to go after hitting this state. When we reach a state with
  // no child and with no additional final weight (i.e. its final
  // weight is the same as its back-off), we can immediately go to its
  // back-off state.
  std::vector<int> next_state_;

  FeatureGroup(const FeatureGroup &) = delete;
  FeatureGroup &operator=(const FeatureGroup &) = delete;
};

template <class A>
struct FeatureGroup<A>::InputOutputLabel {
  Label input, output;

  InputOutputLabel(Label i = kNoLabel, Label o = kNoLabel)
      : input(i), output(o) {}

  bool operator==(InputOutputLabel that) const {
    return input == that.input && output == that.output;
  }

  std::istream &Read(std::istream &strm) {  // NOLINT
    ReadType(strm, &input);
    ReadType(strm, &output);
    return strm;
  }

  std::ostream &Write(std::ostream &strm) const {  // NOLINT
    WriteType(strm, input);
    WriteType(strm, output);
    return strm;
  }
};

template <class A>
struct FeatureGroup<A>::InputOutputLabelHash {
  size_t operator()(InputOutputLabel label) const {
    return static_cast<size_t>(label.input * 7853 + label.output);
  }
};

template <class A>
int FeatureGroup<A>::Walk(int cur, Label ilabel, Label olabel,
                          Weight *weight) const {
  // Note: user of this method need to ensure `ilabel` and `olabel`
  // are valid (e.g. see DCHECKs in
  // `LinearFstData<>::TakeTransition()` and
  // `LinearFstData<>::FindFeature()`).
  int next;
  if (ilabel == LinearFstData<A>::kStartOfSentence) {
    // An observed start-of-sentence only occurs in the beginning of
    // the input, when this feature group is delayed (i.e. there is
    // another feature group with a larger future size). The actual
    // input hasn't arrived so stay at the start state.
    DCHECK_EQ(cur, start_);
    next = start_;
  } else {
    // First, try exact match
    next = FindFirstMatch(InputOutputLabel(ilabel, olabel), cur);
    // Then try with don't cares
    if (next == kNoTrieNodeId)
      next = FindFirstMatch(InputOutputLabel(ilabel, kNoLabel), cur);
    if (next == kNoTrieNodeId)
      next = FindFirstMatch(InputOutputLabel(kNoLabel, olabel), cur);
    // All failed, go to empty context
    if (next == kNoTrieNodeId) next = trie_.Root();
    *weight = Times(*weight, trie_[next].weight);
    next = next_state_[next];
  }
  return next;
}

template <class A>
inline int FeatureGroup<A>::FindFirstMatch(InputOutputLabel label,
                                           int parent) const {
  if (label.input == kNoLabel && label.output == kNoLabel)
    return kNoTrieNodeId;  // very important; see class doc.
  for (; parent != kNoTrieNodeId; parent = trie_[parent].back_link) {
    int next = trie_.Find(parent, label);
    if (next != kNoTrieNodeId) return next;
  }
  return kNoTrieNodeId;
}

template <class A>
inline string FeatureGroup<A>::Stats() const {
  std::ostringstream strm;
  int num_states = 2;
  for (int i = 2; i < next_state_.size(); ++i)
    num_states += i == next_state_[i];
  strm << trie_.NumNodes() << " node(s); " << num_states << " state(s)";
  return strm.str();
}

template <class A>
class LinearFstData<A>::GroupFeatureMap {
 public:
  GroupFeatureMap() {}

  void Init(size_t num_groups, size_t num_words) {
    num_groups_ = num_groups;
    pool_.clear();
    pool_.resize(num_groups * num_words, kNoLabel);
  }

  Label Find(size_t group_id, Label ilabel) const {
    return pool_[IndexOf(group_id, ilabel)];
  }

  bool Set(size_t group_id, Label ilabel, Label feat) {
    size_t i = IndexOf(group_id, ilabel);
    if (pool_[i] != kNoLabel && pool_[i] != feat) {
      FSTERROR() << "Feature group " << group_id
                 << " already has feature for word " << ilabel;
      return false;
    }
    pool_[i] = feat;
    return true;
  }

  std::istream &Read(std::istream &strm) {  // NOLINT
    ReadType(strm, &num_groups_);
    ReadType(strm, &pool_);
    return strm;
  }

  std::ostream &Write(std::ostream &strm) const {  // NOLINT
    WriteType(strm, num_groups_);
    WriteType(strm, pool_);
    return strm;
  }

 private:
  size_t IndexOf(size_t group_id, Label ilabel) const {
    return ilabel * num_groups_ + group_id;
  }

  size_t num_groups_;
  // `pool_[ilabel * num_groups_ + group_id]` is the feature active
  // for group `group_id` with input `ilabel`
  std::vector<Label> pool_;
};

}  // namespace fst

#endif  // FST_EXTENSIONS_LINEAR_LINEAR_FST_DATA_H_
