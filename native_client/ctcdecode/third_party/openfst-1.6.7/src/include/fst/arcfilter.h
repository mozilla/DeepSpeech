// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Function objects to restrict which arcs are traversed in an FST.

#ifndef FST_ARCFILTER_H_
#define FST_ARCFILTER_H_


#include <fst/fst.h>
#include <fst/util.h>


namespace fst {

// True for all arcs.
template <class Arc>
class AnyArcFilter {
 public:
  bool operator()(const Arc &arc) const { return true; }
};

// True for (input/output) epsilon arcs.
template <class Arc>
class EpsilonArcFilter {
 public:
  bool operator()(const Arc &arc) const {
    return arc.ilabel == 0 && arc.olabel == 0;
  }
};

// True for input epsilon arcs.
template <class Arc>
class InputEpsilonArcFilter {
 public:
  bool operator()(const Arc &arc) const { return arc.ilabel == 0; }
};

// True for output epsilon arcs.
template <class Arc>
class OutputEpsilonArcFilter {
 public:
  bool operator()(const Arc &arc) const { return arc.olabel == 0; }
};

// True if specified label matches (doesn't match) when keep_match is
// true (false).
template <class Arc>
class LabelArcFilter {
 public:
  using Label = typename Arc::Label;

  explicit LabelArcFilter(Label label, bool match_input = true,
                          bool keep_match = true)
      : label_(label), match_input_(match_input), keep_match_(keep_match) {}

  bool operator()(const Arc &arc) const {
    const bool match = (match_input_ ? arc.ilabel : arc.olabel) == label_;
    return keep_match_ ? match : !match;
  }

 private:
  const Label label_;
  const bool match_input_;
  const bool keep_match_;
};

// True if specified labels match (don't match) when keep_match is true (false).
template <class Arc>
class MultiLabelArcFilter {
 public:
  using Label = typename Arc::Label;

  explicit MultiLabelArcFilter(bool match_input = true, bool keep_match = true)
      : match_input_(match_input), keep_match_(keep_match) {}

  bool operator()(const Arc &arc) const {
    const Label label = match_input_ ? arc.ilabel : arc.olabel;
    const bool match = labels_.Find(label) != labels_.End();
    return keep_match_ ? match : !match;
  }

  void AddLabel(Label label) { labels_.Insert(label); }

 private:
  CompactSet<Label, kNoLabel> labels_;
  const bool match_input_;
  const bool keep_match_;
};

}  // namespace fst

#endif  // FST_ARCFILTER_H_
