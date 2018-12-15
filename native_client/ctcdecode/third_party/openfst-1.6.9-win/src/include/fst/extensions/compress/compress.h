// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Compresses and decompresses unweighted FSTs.

#ifndef FST_EXTENSIONS_COMPRESS_COMPRESS_H_
#define FST_EXTENSIONS_COMPRESS_COMPRESS_H_

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <fst/compat.h>
#include <fst/log.h>
#include <fst/extensions/compress/elias.h>
#include <fst/extensions/compress/gzfile.h>
#include <fst/encode.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/statesort.h>

namespace fst {

// Identifies stream data as a vanilla compressed FST.
static const int32_t kCompressMagicNumber = 1858869554;
// Identifies stream data as (probably) a Gzip file accidentally read from
// a vanilla stream, without gzip support.
static const int32_t kGzipMagicNumber = 0x8b1f;
// Selects the two most significant bytes.
constexpr uint32_t kGzipMask = 0xffffffff >> 16;

namespace internal {

// Expands a Lempel Ziv code and returns the set of code words. expanded_code[i]
// is the i^th Lempel Ziv codeword.
template <class Var, class Edge>
bool ExpandLZCode(const std::vector<std::pair<Var, Edge>> &code,
                  std::vector<std::vector<Edge>> *expanded_code) {
  expanded_code->resize(code.size());
  for (int i = 0; i < code.size(); ++i) {
    if (code[i].first > i) {
      LOG(ERROR) << "ExpandLZCode: Not a valid code";
      return false;
    }
    if (code[i].first == 0) {
      (*expanded_code)[i].resize(1, code[i].second);
    } else {
      (*expanded_code)[i].resize((*expanded_code)[code[i].first - 1].size() +
                                 1);
      std::copy((*expanded_code)[code[i].first - 1].begin(),
                (*expanded_code)[code[i].first - 1].end(),
                (*expanded_code)[i].begin());
      (*expanded_code)[i][(*expanded_code)[code[i].first - 1].size()] =
          code[i].second;
    }
  }
  return true;
}

}  // namespace internal

// Lempel Ziv on data structure Edge, with a less than operator
// EdgeLessThan and an equals operator  EdgeEquals.
// Edge has a value defaultedge which it never takes and
// Edge is defined, it is initialized to defaultedge
template <class Var, class Edge, class EdgeLessThan, class EdgeEquals>
class LempelZiv {
 public:
  LempelZiv() : dict_number_(0), default_edge_() {
    root_.current_number = dict_number_++;
    root_.current_edge = default_edge_;
    decode_vector_.push_back(std::make_pair(0, default_edge_));
  }
  // Encodes a vector input into output
  void BatchEncode(const std::vector<Edge> &input,
                   std::vector<std::pair<Var, Edge>> *output);

  // Decodes codedvector to output. Returns false if
  // the index exceeds the size.
  bool BatchDecode(const std::vector<std::pair<Var, Edge>> &input,
                   std::vector<Edge> *output);

  // Decodes a single dictionary element. Returns false
  // if the index exceeds the size.
  bool SingleDecode(const Var &index, Edge *output) {
    if (index >= decode_vector_.size()) {
      LOG(ERROR) << "LempelZiv::SingleDecode: "
                 << "Index exceeded the dictionary size";
      return false;
    } else {
      *output = decode_vector_[index].second;
      return true;
    }
  }

  ~LempelZiv() {
    for (auto it = (root_.next_number).begin(); it != (root_.next_number).end();
         ++it) {
      CleanUp(it->second);
    }
  }
  // Adds a single dictionary element while decoding
  //  void AddDictElement(const std::pair<Var, Edge> &newdict) {
  //    EdgeEquals InstEdgeEquals;
  //  if (InstEdgeEquals(newdict.second, default_edge_) != 1)
  //     decode_vector_.push_back(newdict);
  //  }

 private:
  // Node datastructure is used for encoding

  struct Node {
    Var current_number;
    Edge current_edge;
    std::map<Edge, Node *, EdgeLessThan> next_number;
  };

  void CleanUp(Node *temp) {
    for (auto it = (temp->next_number).begin(); it != (temp->next_number).end();
         ++it) {
      CleanUp(it->second);
    }
    delete temp;
  }
  Node root_;
  Var dict_number_;
  // decode_vector_ is used for decoding
  std::vector<std::pair<Var, Edge>> decode_vector_;
  Edge default_edge_;
};

template <class Var, class Edge, class EdgeLessThan, class EdgeEquals>
void LempelZiv<Var, Edge, EdgeLessThan, EdgeEquals>::BatchEncode(
    const std::vector<Edge> &input, std::vector<std::pair<Var, Edge>> *output) {
  for (typename std::vector<Edge>::const_iterator it = input.begin();
       it != input.end(); ++it) {
    Node *temp_node = &root_;
    while (it != input.end()) {
      auto next = (temp_node->next_number).find(*it);
      if (next != (temp_node->next_number).end()) {
        temp_node = next->second;
        ++it;
      } else {
        break;
      }
    }
    if (it == input.end() && temp_node->current_number != 0) {
      output->push_back(
          std::make_pair(temp_node->current_number, default_edge_));
    } else if (it != input.end()) {
      output->push_back(std::make_pair(temp_node->current_number, *it));
      Node *new_node = new (Node);
      new_node->current_number = dict_number_++;
      new_node->current_edge = *it;
      (temp_node->next_number)[*it] = new_node;
    }
    if (it == input.end()) break;
  }
}

template <class Var, class Edge, class EdgeLessThan, class EdgeEquals>
bool LempelZiv<Var, Edge, EdgeLessThan, EdgeEquals>::BatchDecode(
    const std::vector<std::pair<Var, Edge>> &input, std::vector<Edge> *output) {
  for (typename std::vector<std::pair<Var, Edge>>::const_iterator it =
           input.begin();
       it != input.end(); ++it) {
    std::vector<Edge> temp_output;
    EdgeEquals InstEdgeEquals;
    if (InstEdgeEquals(it->second, default_edge_) != 1) {
      decode_vector_.push_back(*it);
      temp_output.push_back(it->second);
    }
    Var temp_integer = it->first;
    if (temp_integer >= decode_vector_.size()) {
      LOG(ERROR) << "LempelZiv::BatchDecode: "
                 << "Index exceeded the dictionary size";
      return false;
    } else {
      while (temp_integer != 0) {
        temp_output.push_back(decode_vector_[temp_integer].second);
        temp_integer = decode_vector_[temp_integer].first;
      }
      std::reverse(temp_output.begin(), temp_output.end());
      output->insert(output->end(), temp_output.begin(), temp_output.end());
    }
  }
  return true;
}

// The main Compressor class
template <class Arc>
class Compressor {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  Compressor() {}

  // Compresses fst into a boolean vector code. Returns true on sucesss.
  bool Compress(const Fst<Arc> &fst, std::ostream &strm);

  // Decompresses the boolean vector into Fst. Returns true on sucesss.
  bool Decompress(std::istream &strm, const string &source,
                  MutableFst<Arc> *fst);

  // Finds the BFS order of a fst
  void BfsOrder(const ExpandedFst<Arc> &fst, std::vector<StateId> *order);

  // Preprocessing step to convert fst to a isomorphic fst
  // Returns a preproccess fst and a dictionary
  void Preprocess(const Fst<Arc> &fst, MutableFst<Arc> *preprocessedfst,
                  EncodeMapper<Arc> *encoder);

  // Performs Lempel Ziv and outputs a stream of integers
  // and sends it to a stream
  void EncodeProcessedFst(const ExpandedFst<Arc> &fst, std::ostream &strm);

  // Decodes fst from the stream
  void DecodeProcessedFst(const std::vector<StateId> &input,
                          MutableFst<Arc> *fst, bool unweighted);

  // Converts buffer_code_ to uint8_t and writes to a stream.

  // Writes the boolean file to the stream
  void WriteToStream(std::ostream &strm);

  // Writes the weights to the stream
  void WriteWeight(const std::vector<Weight> &input, std::ostream &strm);

  void ReadWeight(std::istream &strm, std::vector<Weight> *output);

  // Same as fst::Decode without the line RmFinalEpsilon(fst)
  void DecodeForCompress(MutableFst<Arc> *fst, const EncodeMapper<Arc> &mapper);

  // Updates the buffer_code_
  template <class CVar>
  void WriteToBuffer(CVar input) {
    std::vector<bool> current_code;
    Elias<CVar>::DeltaEncode(input, &current_code);
    if (!buffer_code_.empty()) {
      buffer_code_.insert(buffer_code_.end(), current_code.begin(),
                          current_code.end());
    } else {
      buffer_code_.assign(current_code.begin(), current_code.end());
    }
  }

 private:
  struct LZLabel {
    LZLabel() : label(0) {}
    Label label;
  };

  struct LabelLessThan {
    bool operator()(const LZLabel &labelone, const LZLabel &labeltwo) const {
      return labelone.label < labeltwo.label;
    }
  };

  struct LabelEquals {
    bool operator()(const LZLabel &labelone, const LZLabel &labeltwo) const {
      return labelone.label == labeltwo.label;
    }
  };

  struct Transition {
    Transition() : nextstate(0), label(0), weight(Weight::Zero()) {}

    StateId nextstate;
    Label label;
    Weight weight;
  };

  struct TransitionLessThan {
    bool operator()(const Transition &transition_one,
                    const Transition &transition_two) const {
      if (transition_one.nextstate == transition_two.nextstate)
        return transition_one.label < transition_two.label;
      else
        return transition_one.nextstate < transition_two.nextstate;
    }
  } transition_less_than;

  struct TransitionEquals {
    bool operator()(const Transition &transition_one,
                    const Transition &transition_two) const {
      return transition_one.nextstate == transition_two.nextstate &&
             transition_one.label == transition_two.label;
    }
  } transition_equals;

  struct OldDictCompare {
    bool operator()(const std::pair<StateId, Transition> &pair_one,
                    const std::pair<StateId, Transition> &pair_two) const {
      if ((pair_one.second).nextstate == (pair_two.second).nextstate)
        return (pair_one.second).label < (pair_two.second).label;
      else
        return (pair_one.second).nextstate < (pair_two.second).nextstate;
    }
  } old_dict_compare;

  std::vector<bool> buffer_code_;
  std::vector<Weight> arc_weight_;
  std::vector<Weight> final_weight_;
};

template <class Arc>
inline void Compressor<Arc>::DecodeForCompress(
    MutableFst<Arc> *fst, const EncodeMapper<Arc> &mapper) {
  ArcMap(fst, EncodeMapper<Arc>(mapper, DECODE));
  fst->SetInputSymbols(mapper.InputSymbols());
  fst->SetOutputSymbols(mapper.OutputSymbols());
}

// Compressor::BfsOrder
template <class Arc>
void Compressor<Arc>::BfsOrder(const ExpandedFst<Arc> &fst,
                               std::vector<StateId> *order) {
  Arc arc;
  StateId bfs_visit_number = 0;
  std::queue<StateId> states_queue;
  order->assign(fst.NumStates(), kNoStateId);
  states_queue.push(fst.Start());
  (*order)[fst.Start()] = bfs_visit_number++;
  while (!states_queue.empty()) {
    for (ArcIterator<Fst<Arc>> aiter(fst, states_queue.front()); !aiter.Done();
         aiter.Next()) {
      arc = aiter.Value();
      StateId nextstate = arc.nextstate;
      if ((*order)[nextstate] == kNoStateId) {
        (*order)[nextstate] = bfs_visit_number++;
        states_queue.push(nextstate);
      }
    }
    states_queue.pop();
  }

  // If the FST is unconnected, then the following
  // code finds them
  while (bfs_visit_number < fst.NumStates()) {
    int unseen_state = 0;
    for (unseen_state = 0; unseen_state < fst.NumStates(); ++unseen_state) {
      if ((*order)[unseen_state] == kNoStateId) break;
    }
    states_queue.push(unseen_state);
    (*order)[unseen_state] = bfs_visit_number++;
    while (!states_queue.empty()) {
      for (ArcIterator<Fst<Arc>> aiter(fst, states_queue.front());
           !aiter.Done(); aiter.Next()) {
        arc = aiter.Value();
        StateId nextstate = arc.nextstate;
        if ((*order)[nextstate] == kNoStateId) {
          (*order)[nextstate] = bfs_visit_number++;
          states_queue.push(nextstate);
        }
      }
      states_queue.pop();
    }
  }
}

template <class Arc>
void Compressor<Arc>::Preprocess(const Fst<Arc> &fst,
                                 MutableFst<Arc> *preprocessedfst,
                                 EncodeMapper<Arc> *encoder) {
  *preprocessedfst = fst;
  if (!preprocessedfst->NumStates()) {
    return;
  }
  // Relabels the edges and develops a dictionary
  Encode(preprocessedfst, encoder);
  std::vector<StateId> order;
  // Finds the BFS sorting order of the fst
  BfsOrder(*preprocessedfst, &order);
  // Reorders the states according to the BFS order
  StateSort(preprocessedfst, order);
}

template <class Arc>
void Compressor<Arc>::EncodeProcessedFst(const ExpandedFst<Arc> &fst,
                                         std::ostream &strm) {
  std::vector<StateId> output;
  LempelZiv<StateId, LZLabel, LabelLessThan, LabelEquals> dict_new;
  LempelZiv<StateId, Transition, TransitionLessThan, TransitionEquals> dict_old;
  std::vector<LZLabel> current_new_input;
  std::vector<Transition> current_old_input;
  std::vector<std::pair<StateId, LZLabel>> current_new_output;
  std::vector<std::pair<StateId, Transition>> current_old_output;
  std::vector<StateId> final_states;

  StateId number_of_states = fst.NumStates();

  StateId seen_states = 0;
  // Adding the number of states
  WriteToBuffer<StateId>(number_of_states);

  for (StateId state = 0; state < number_of_states; ++state) {
    current_new_input.clear();
    current_old_input.clear();
    current_new_output.clear();
    current_old_output.clear();
    if (state > seen_states) ++seen_states;

    // Collecting the final states
    if (fst.Final(state) != Weight::Zero()) {
      final_states.push_back(state);
      final_weight_.push_back(fst.Final(state));
    }

    // Reading the states
    for (ArcIterator<Fst<Arc>> aiter(fst, state); !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.nextstate > seen_states) {  // RILEY: > or >= ?
        ++seen_states;
        LZLabel temp_label;
        temp_label.label = arc.ilabel;
        arc_weight_.push_back(arc.weight);
        current_new_input.push_back(temp_label);
      } else {
        Transition temp_transition;
        temp_transition.nextstate = arc.nextstate;
        temp_transition.label = arc.ilabel;
        temp_transition.weight = arc.weight;
        current_old_input.push_back(temp_transition);
      }
    }
    // Adding new states
    dict_new.BatchEncode(current_new_input, &current_new_output);
    WriteToBuffer<StateId>(current_new_output.size());

    for (auto it = current_new_output.begin(); it != current_new_output.end();
         ++it) {
      WriteToBuffer<StateId>(it->first);
      WriteToBuffer<Label>((it->second).label);
    }
    // Adding old states by sorting and using difference coding
    std::sort(current_old_input.begin(), current_old_input.end(),
              transition_less_than);
    for (auto it = current_old_input.begin(); it != current_old_input.end();
         ++it) {
      arc_weight_.push_back(it->weight);
    }
    dict_old.BatchEncode(current_old_input, &current_old_output);
    std::vector<StateId> dict_old_temp;
    std::vector<Transition> transition_old_temp;
    for (auto it = current_old_output.begin(); it != current_old_output.end();
         ++it) {
      dict_old_temp.push_back(it->first);
      transition_old_temp.push_back(it->second);
    }
    if (!transition_old_temp.empty()) {
      if ((transition_old_temp.back()).nextstate == 0 &&
          (transition_old_temp.back()).label == 0) {
        transition_old_temp.pop_back();
      }
    }
    std::sort(dict_old_temp.begin(), dict_old_temp.end());
    std::sort(transition_old_temp.begin(), transition_old_temp.end(),
              transition_less_than);

    WriteToBuffer<StateId>(dict_old_temp.size());
    if (dict_old_temp.size() != transition_old_temp.size())
      WriteToBuffer<int>(1);
    else
      WriteToBuffer<int>(0);

    StateId previous;
    if (!dict_old_temp.empty()) {
      WriteToBuffer<StateId>(dict_old_temp.front());
      previous = dict_old_temp.front();
    }
    if (dict_old_temp.size() > 1) {
      for (auto it = dict_old_temp.begin() + 1; it != dict_old_temp.end();
           ++it) {
        WriteToBuffer<StateId>(*it - previous);
        previous = *it;
      }
    }
    if (!transition_old_temp.empty()) {
      WriteToBuffer<StateId>((transition_old_temp.front()).nextstate);
      previous = (transition_old_temp.front()).nextstate;
      WriteToBuffer<Label>((transition_old_temp.front()).label);
    }
    if (transition_old_temp.size() > 1) {
      for (auto it = transition_old_temp.begin() + 1;
           it != transition_old_temp.end(); ++it) {
        WriteToBuffer<StateId>(it->nextstate - previous);
        previous = it->nextstate;
        WriteToBuffer<StateId>(it->label);
      }
    }
  }
  // Adding final states
  WriteToBuffer<StateId>(final_states.size());
  if (!final_states.empty()) {
    for (auto it = final_states.begin(); it != final_states.end(); ++it) {
      WriteToBuffer<StateId>(*it);
    }
  }
  WriteToStream(strm);
  uint8_t unweighted = (fst.Properties(kUnweighted, true) == kUnweighted);
  WriteType(strm, unweighted);
  if (unweighted == 0) {
    WriteWeight(arc_weight_, strm);
    WriteWeight(final_weight_, strm);
  }
}

template <class Arc>
void Compressor<Arc>::DecodeProcessedFst(const std::vector<StateId> &input,
                                         MutableFst<Arc> *fst,
                                         bool unweighted) {
  LempelZiv<StateId, LZLabel, LabelLessThan, LabelEquals> dict_new;
  LempelZiv<StateId, Transition, TransitionLessThan, TransitionEquals> dict_old;
  std::vector<std::pair<StateId, LZLabel>> current_new_input;
  std::vector<std::pair<StateId, Transition>> current_old_input;
  std::vector<LZLabel> current_new_output;
  std::vector<Transition> current_old_output;
  std::vector<std::pair<StateId, Transition>> actual_old_dict_numbers;
  std::vector<Transition> actual_old_dict_transitions;
  auto arc_weight_it = arc_weight_.begin();
  Transition default_transition;
  StateId seen_states = 1;

  // Adding states.
  const StateId num_states = input.front();
  if (num_states > 0) {
    const StateId start_state = fst->AddState();
    fst->SetStart(start_state);
    for (StateId state = 1; state < num_states; ++state) {
      fst->AddState();
    }
  }

  typename std::vector<StateId>::const_iterator main_it = input.begin();
  ++main_it;

  for (StateId current_state = 0; current_state < num_states; ++current_state) {
    if (current_state >= seen_states) ++seen_states;
    current_new_input.clear();
    current_new_output.clear();
    current_old_input.clear();
    current_old_output.clear();
    // New states
    StateId current_number_new_elements = *main_it;
    ++main_it;
    for (StateId new_integer = 0; new_integer < current_number_new_elements;
         ++new_integer) {
      std::pair<StateId, LZLabel> temp_new_dict_element;
      temp_new_dict_element.first = *main_it;
      ++main_it;
      LZLabel temp_label;
      temp_label.label = *main_it;
      ++main_it;
      temp_new_dict_element.second = temp_label;
      current_new_input.push_back(temp_new_dict_element);
    }
    dict_new.BatchDecode(current_new_input, &current_new_output);
    for (auto it = current_new_output.begin(); it != current_new_output.end();
         ++it) {
      if (!unweighted) {
        fst->AddArc(current_state,
                    Arc(it->label, it->label, *arc_weight_it, seen_states++));
        ++arc_weight_it;
      } else {
        fst->AddArc(current_state,
                    Arc(it->label, it->label, Weight::One(), seen_states++));
      }
    }

    // Old states dictionary
    StateId current_number_old_elements = *main_it;
    ++main_it;
    StateId is_zero_removed = *main_it;
    ++main_it;
    StateId previous = 0;
    actual_old_dict_numbers.clear();
    for (StateId new_integer = 0; new_integer < current_number_old_elements;
         ++new_integer) {
      std::pair<StateId, Transition> pair_temp_transition;
      if (new_integer == 0) {
        pair_temp_transition.first = *main_it;
        previous = *main_it;
      } else {
        pair_temp_transition.first = *main_it + previous;
        previous = pair_temp_transition.first;
      }
      ++main_it;
      Transition temp_test;
      if (!dict_old.SingleDecode(pair_temp_transition.first, &temp_test)) {
        FSTERROR() << "Compressor::Decode: failed";
        fst->DeleteStates();
        fst->SetProperties(kError, kError);
        return;
      }
      pair_temp_transition.second = temp_test;
      actual_old_dict_numbers.push_back(pair_temp_transition);
    }

    // Reordering the dictionary elements
    std::sort(actual_old_dict_numbers.begin(), actual_old_dict_numbers.end(),
              old_dict_compare);

    // Transitions
    previous = 0;
    actual_old_dict_transitions.clear();

    for (StateId new_integer = 0;
         new_integer < current_number_old_elements - is_zero_removed;
         ++new_integer) {
      Transition temp_transition;
      if (new_integer == 0) {
        temp_transition.nextstate = *main_it;
        previous = *main_it;
      } else {
        temp_transition.nextstate = *main_it + previous;
        previous = temp_transition.nextstate;
      }
      ++main_it;
      temp_transition.label = *main_it;
      ++main_it;
      actual_old_dict_transitions.push_back(temp_transition);
    }

    if (is_zero_removed == 1) {
      actual_old_dict_transitions.push_back(default_transition);
    }

    auto trans_it = actual_old_dict_transitions.begin();
    auto dict_it = actual_old_dict_numbers.begin();

    while (trans_it != actual_old_dict_transitions.end() &&
           dict_it != actual_old_dict_numbers.end()) {
      if (dict_it->first == 0) {
        ++dict_it;
      } else {
        std::pair<StateId, Transition> temp_pair;
        if (transition_equals(*trans_it, default_transition) == 1) {
          temp_pair.first = dict_it->first;
          temp_pair.second = default_transition;
          ++dict_it;
        } else if (transition_less_than(dict_it->second, *trans_it) == 1) {
          temp_pair.first = dict_it->first;
          temp_pair.second = *trans_it;
          ++dict_it;
        } else {
          temp_pair.first = 0;
          temp_pair.second = *trans_it;
        }
        ++trans_it;
        current_old_input.push_back(temp_pair);
      }
    }
    while (trans_it != actual_old_dict_transitions.end()) {
      std::pair<StateId, Transition> temp_pair;
      temp_pair.first = 0;
      temp_pair.second = *trans_it;
      ++trans_it;
      current_old_input.push_back(temp_pair);
    }

    // Adding old elements in the dictionary
    if (!dict_old.BatchDecode(current_old_input, &current_old_output)) {
      FSTERROR() << "Compressor::Decode: Failed";
      fst->DeleteStates();
      fst->SetProperties(kError, kError);
      return;
    }

    for (auto it = current_old_output.begin(); it != current_old_output.end();
         ++it) {
      if (!unweighted) {
        fst->AddArc(current_state,
                    Arc(it->label, it->label, *arc_weight_it, it->nextstate));
        ++arc_weight_it;
      } else {
        fst->AddArc(current_state,
                    Arc(it->label, it->label, Weight::One(), it->nextstate));
      }
    }
  }
  // Adding the final states
  StateId number_of_final_states = *main_it;
  if (number_of_final_states > 0) {
    ++main_it;
    for (StateId temp_int = 0; temp_int < number_of_final_states; ++temp_int) {
      if (!unweighted) {
        fst->SetFinal(*main_it, final_weight_[temp_int]);
      } else {
        fst->SetFinal(*main_it, Weight(0));
      }
      ++main_it;
    }
  }
}

template <class Arc>
void Compressor<Arc>::ReadWeight(std::istream &strm,
                                 std::vector<Weight> *output) {
  int64_t size;
  Weight weight;
  ReadType(strm, &size);
  for (int64_t i = 0; i < size; ++i) {
    weight.Read(strm);
    output->push_back(weight);
  }
}

template <class Arc>
bool Compressor<Arc>::Decompress(std::istream &strm, const string &source,
                                 MutableFst<Arc> *fst) {
  fst->DeleteStates();
  int32_t magic_number = 0;
  ReadType(strm, &magic_number);
  if (magic_number != kCompressMagicNumber) {
    LOG(ERROR) << "Decompress: Bad compressed Fst: " << source;
    // If the most significant two bytes of the magic number match the
    // gzip magic number, then we are probably reading a gzip file as an
    // ordinary stream.
    if ((magic_number & kGzipMask) == kGzipMagicNumber) {
      LOG(ERROR) << "Decompress: Fst appears to be compressed with Gzip, but "
                    "gzip decompression was not requested. Try with "
                    "the --gzip flag"
                    ".";
    }
    return false;
  }
  std::unique_ptr<EncodeMapper<Arc>> encoder(
      EncodeMapper<Arc>::Read(strm, "Decoding", DECODE));
  std::vector<bool> bool_code;
  uint8_t block;
  uint8_t msb = 128;
  int64_t data_size;
  ReadType(strm, &data_size);
  for (int64_t i = 0; i < data_size; ++i) {
    ReadType(strm, &block);
    for (int j = 0; j < 8; ++j) {
      uint8_t temp = msb & block;
      if (temp == 128)
        bool_code.push_back(1);
      else
        bool_code.push_back(0);
      block = block << 1;
    }
  }
  std::vector<StateId> int_code;
  Elias<StateId>::BatchDecode(bool_code, &int_code);
  bool_code.clear();
  uint8_t unweighted;
  ReadType(strm, &unweighted);
  if (unweighted == 0) {
    ReadWeight(strm, &arc_weight_);
    ReadWeight(strm, &final_weight_);
  }
  DecodeProcessedFst(int_code, fst, unweighted);
  DecodeForCompress(fst, *encoder);
  return !fst->Properties(kError, false);
}

template <class Arc>
void Compressor<Arc>::WriteWeight(const std::vector<Weight> &input,
                                  std::ostream &strm) {
  int64_t size = input.size();
  WriteType(strm, size);
  for (typename std::vector<Weight>::const_iterator it = input.begin();
       it != input.end(); ++it) {
    it->Write(strm);
  }
}

template <class Arc>
void Compressor<Arc>::WriteToStream(std::ostream &strm) {
  while (buffer_code_.size() % 8 != 0) buffer_code_.push_back(1);
  int64_t data_size = buffer_code_.size() / 8;
  WriteType(strm, data_size);
  std::vector<bool>::const_iterator it;
  int64_t i;
  uint8_t block;
  for (it = buffer_code_.begin(), i = 0; it != buffer_code_.end(); ++it, ++i) {
    if (i % 8 == 0) {
      if (i > 0) WriteType(strm, block);
      block = 0;
    } else {
      block = block << 1;
    }
    block |= *it;
  }
  WriteType(strm, block);
}

template <class Arc>
bool Compressor<Arc>::Compress(const Fst<Arc> &fst, std::ostream &strm) {
  VectorFst<Arc> processedfst;
  EncodeMapper<Arc> encoder(kEncodeLabels, ENCODE);
  Preprocess(fst, &processedfst, &encoder);
  WriteType(strm, kCompressMagicNumber);
  encoder.Write(strm, "encoder stream");
  EncodeProcessedFst(processedfst, strm);
  return true;
}

// Convenience functions that call the compressor and decompressor.

template <class Arc>
void Compress(const Fst<Arc> &fst, std::ostream &strm) {
  Compressor<Arc> comp;
  comp.Compress(fst, strm);
}

// Returns true on success.
template <class Arc>
bool Compress(const Fst<Arc> &fst, const string &file_name,
              const bool gzip = false) {
  if (gzip) {
    if (file_name.empty()) {
      std::stringstream strm;
      Compress(fst, strm);
      OGzFile gzfile(fileno(stdout));
      gzfile.write(strm);
      if (!gzfile) {
        LOG(ERROR) << "Compress: Can't write to file: stdout";
        return false;
      }
    } else {
      std::stringstream strm;
      Compress(fst, strm);
      OGzFile gzfile(file_name);
      if (!gzfile) {
        LOG(ERROR) << "Compress: Can't open file: " << file_name;
        return false;
      }
      gzfile.write(strm);
      if (!gzfile) {
        LOG(ERROR) << "Compress: Can't write to file: " << file_name;
        return false;
      }
    }
  } else if (file_name.empty()) {
    Compress(fst, std::cout);
  } else {
    std::ofstream strm(file_name,
                             std::ios_base::out | std::ios_base::binary);
    if (!strm) {
      LOG(ERROR) << "Compress: Can't open file: " << file_name;
      return false;
    }
    Compress(fst, strm);
  }
  return true;
}

template <class Arc>
void Decompress(std::istream &strm, const string &source,
                MutableFst<Arc> *fst) {
  Compressor<Arc> comp;
  comp.Decompress(strm, source, fst);
}

// Returns true on success.
template <class Arc>
bool Decompress(const string &file_name, MutableFst<Arc> *fst,
                const bool gzip = false) {
  if (gzip) {
    if (file_name.empty()) {
      IGzFile gzfile(fileno(stdin));
      Decompress(*gzfile.read(), "stdin", fst);
      if (!gzfile) {
        LOG(ERROR) << "Decompress: Can't read from file: stdin";
        return false;
      }
    } else {
      IGzFile gzfile(file_name);
      if (!gzfile) {
        LOG(ERROR) << "Decompress: Can't open file: " << file_name;
        return false;
      }
      Decompress(*gzfile.read(), file_name, fst);
      if (!gzfile) {
        LOG(ERROR) << "Decompress: Can't read from file: " << file_name;
        return false;
      }
    }
  } else if (file_name.empty()) {
    Decompress(std::cin, "stdin", fst);
  } else {
    std::ifstream strm(file_name,
                            std::ios_base::in | std::ios_base::binary);
    if (!strm) {
      LOG(ERROR) << "Decompress: Can't open file: " << file_name;
      return false;
    }
    Decompress(strm, file_name, fst);
  }
  return true;
}

}  // namespace fst

#endif  // FST_EXTENSIONS_COMPRESS_COMPRESS_H_
