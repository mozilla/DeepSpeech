/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This test illustrates how to make use of the CTCBeamSearchDecoder using a
// custom BeamScorer and BeamState based on a dictionary with a few artificial
// words.
#include "tensorflow/core/util/ctc/ctc_beam_search.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "kenlm/lm/model.hh"

#include "alphabet.h"
#include "trie_node.h"

namespace tf = tensorflow;
using tf::shape_inference::DimensionHandle;
using tf::shape_inference::InferenceContext;
using tf::shape_inference::ShapeHandle;

REGISTER_OP("CTCBeamSearchDecoderWithLM")
    .Input("inputs: float")
    .Input("sequence_length: int32")
    .Attr("model_path: string")
    .Attr("trie_path: string")
    .Attr("alphabet_path: string")
    .Attr("beam_width: int >= 1 = 100")
    .Attr("top_paths: int >= 1 = 1")
    .Attr("merge_repeated: bool = true")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("log_probability: float")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

      // Get batch size from inputs and sequence_length.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));

      tf::int32 top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));

      // Outputs.
      int out_idx = 0;
      for (int i = 0; i < top_paths; ++i) {  // decoded_indices
        c->set_output(out_idx++, c->Matrix(InferenceContext::kUnknownDim, 2));
      }
      for (int i = 0; i < top_paths; ++i) {  // decoded_values
        c->set_output(out_idx++, c->Vector(InferenceContext::kUnknownDim));
      }
      ShapeHandle shape_v = c->Vector(2);
      for (int i = 0; i < top_paths; ++i) {  // decoded_shape
        c->set_output(out_idx++, shape_v);
      }
      c->set_output(out_idx++, c->Matrix(batch_size, top_paths));
      return tf::Status::OK();
    })
    .Doc(R"doc(
Performs beam search decoding on the logits given in input.

A note about the attribute merge_repeated: For the beam search decoder,
this means that if consecutive entries in a beam are the same, only
the first of these is emitted.  That is, when the top path is "A B B B B",
"A B" is returned if merge_repeated = True but "A B B B B" is
returned if merge_repeated = False.

inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
sequence_length: A vector containing sequence lengths, size `(batch)`.
beam_width: A scalar >= 0 (beam search beam width).
top_paths: A scalar >= 0, <= beam_width (controls output size).
merge_repeated: If true, merge repeated classes in output.
decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,
  size `(total_decoded_outputs[j] x 2)`, has indices of a
  `SparseTensor<int64, 2>`.  The rows store: [batch, time].
decoded_values: A list (length: top_paths) of values vectors.  Vector j,
  size `(length total_decoded_outputs[j])`, has the values of a
  `SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
decoded_shape: A list (length: top_paths) of shape vector.  Vector j,
  size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
  Its values are: `[batch_size, max_decoded_length[j]]`.
log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The
  sequence log-probabilities.
)doc");

struct KenLMBeamState {
  float language_model_score;
  float score;
  float delta_score;
  std::string incomplete_word;
  TrieNode *incomplete_word_trie_node;
  lm::ngram::ProbingModel::State model_state;
};

class KenLMBeamScorer : public tf::ctc::BaseBeamScorer<KenLMBeamState> {
 public:
  typedef lm::ngram::ProbingModel Model;

  KenLMBeamScorer(const std::string &kenlm_path, const std::string &trie_path, const std::string &alphabet_path)
    : lm_weight(1.0f)
    , word_count_weight(-0.1f)
    , valid_word_count_weight(1.0f)
  {
    lm::ngram::Config config;
    config.load_method = util::POPULATE_OR_READ;
    model = new Model(kenlm_path.c_str(), config);

    alphabet = new Alphabet(alphabet_path.c_str());

    std::ifstream in;
    in.open(trie_path, std::ios::in);
    TrieNode::ReadFromStream(in, trieRoot, alphabet->GetSize());
    in.close();
  }

  virtual ~KenLMBeamScorer() {
    delete model;
    delete trieRoot;
  }

  // State initialization.
  void InitializeState(KenLMBeamState* root) const {
    root->language_model_score = 0.0f;
    root->score = 0.0f;
    root->delta_score = 0.0f;
    root->incomplete_word.clear();
    root->incomplete_word_trie_node = trieRoot;
    root->model_state = model->BeginSentenceState();
  }
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  void ExpandState(const KenLMBeamState& from_state, int from_label,
                         KenLMBeamState* to_state, int to_label) const {
    CopyState(from_state, to_state);

    if (!alphabet->IsSpace(to_label)) {
      to_state->incomplete_word += alphabet->StringFromLabel(to_label);
      TrieNode *trie_node = from_state.incomplete_word_trie_node;

      // TODO replace with OOV unigram prob?
      // If we have no valid prefix we assume a very low log probability
      float min_unigram_score = -10.0f;
      // If prefix does exist
      if (trie_node != nullptr) {
        trie_node = trie_node->GetChildAt(to_label);
        to_state->incomplete_word_trie_node = trie_node;

        if (trie_node != nullptr) {
          min_unigram_score = trie_node->GetMinUnigramScore();
        }
      }
      // TODO try two options
      // 1) unigram score added up to language model scare
      // 2) langugage model score of (preceding_words + unigram_word)
      to_state->score = min_unigram_score + to_state->language_model_score;
      to_state->delta_score = to_state->score - from_state.score;
    } else {
      float lm_score_delta = ScoreIncompleteWord(from_state.model_state,
                            to_state->incomplete_word,
                            to_state->model_state);
      // Give fixed word bonus
      if (!IsOOV(to_state->incomplete_word)) {
        to_state->language_model_score += valid_word_count_weight;
      }
      to_state->language_model_score += word_count_weight;
      UpdateWithLMScore(to_state, lm_score_delta);
      ResetIncompleteWord(to_state);
    }
  }
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  void ExpandStateEnd(KenLMBeamState* state) const {
    float lm_score_delta = 0.0f;
    Model::State out;
    if (state->incomplete_word.size() > 0) {
      lm_score_delta += ScoreIncompleteWord(state->model_state,
                                            state->incomplete_word,
                                            out);
      ResetIncompleteWord(state);
      state->model_state = out;
    }
    lm_score_delta += model->FullScore(state->model_state,
                                      model->GetVocabulary().EndSentence(),
                                      out).prob;
    UpdateWithLMScore(state, lm_score_delta);
  }
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  float GetStateExpansionScore(const KenLMBeamState& state,
                               float previous_score) const {
    return lm_weight * state.delta_score + previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  float GetStateEndExpansionScore(const KenLMBeamState& state) const {
    return lm_weight * state.delta_score;
  }

  void SetLMWeight(float lm_weight) {
    this->lm_weight = lm_weight;
  }

  void SetWordCountWeight(float word_count_weight) {
    this->word_count_weight = word_count_weight;
  }

  void SetValidWordCountWeight(float valid_word_count_weight) {
    this->valid_word_count_weight = valid_word_count_weight;
  }

 private:
  Model *model;
  Alphabet *alphabet;
  TrieNode *trieRoot;
  float lm_weight;
  float word_count_weight;
  float valid_word_count_weight;

  void UpdateWithLMScore(KenLMBeamState *state, float lm_score_delta) const {
    float previous_score = state->score;
    state->language_model_score += lm_score_delta;
    state->score = state->language_model_score;
    state->delta_score = state->language_model_score - previous_score;
  }

  void ResetIncompleteWord(KenLMBeamState *state) const {
    state->incomplete_word.clear();
    state->incomplete_word_trie_node = trieRoot;
  }

  bool IsOOV(const std::string& word) const {
    auto &vocabulary = model->GetVocabulary();
    return vocabulary.Index(word) == vocabulary.NotFound();
  }

  float ScoreIncompleteWord(const Model::State& model_state,
                            const std::string& word,
                            Model::State& out) const {
    lm::FullScoreReturn full_score_return;
    lm::WordIndex vocab = model->GetVocabulary().Index(word);
    full_score_return = model->FullScore(model_state, vocab, out);
    return full_score_return.prob;
  }

  void CopyState(const KenLMBeamState& from, KenLMBeamState* to) const {
    to->language_model_score = from.language_model_score;
    to->score = from.score;
    to->delta_score = from.delta_score;
    to->incomplete_word = from.incomplete_word;
    to->incomplete_word_trie_node = from.incomplete_word_trie_node;
    to->model_state = from.model_state;
  }
};

class CTCDecodeHelper {
 public:
  CTCDecodeHelper() : top_paths_(1) {}

  inline int GetTopPaths() const { return top_paths_; }
  void SetTopPaths(int tp) { top_paths_ = tp; }

  tf::Status ValidateInputsGenerateOutputs(
      tf::OpKernelContext *ctx, const tf::Tensor **inputs, const tf::Tensor **seq_len,
      std::string *model_path, std::string *trie_path, std::string *alphabet_path,
      tf::Tensor **log_prob, tf::OpOutputList *decoded_indices,
      tf::OpOutputList *decoded_values, tf::OpOutputList *decoded_shape) const {
    tf::Status status = ctx->input("inputs", inputs);
    if (!status.ok()) return status;
    status = ctx->input("sequence_length", seq_len);
    if (!status.ok()) return status;

    const tf::TensorShape &inputs_shape = (*inputs)->shape();

    if (inputs_shape.dims() != 3) {
      return tf::errors::InvalidArgument("inputs is not a 3-Tensor");
    }

    const tf::int64 max_time = inputs_shape.dim_size(0);
    const tf::int64 batch_size = inputs_shape.dim_size(1);

    if (max_time == 0) {
      return tf::errors::InvalidArgument("max_time is 0");
    }
    if (!tf::TensorShapeUtils::IsVector((*seq_len)->shape())) {
      return tf::errors::InvalidArgument("sequence_length is not a vector");
    }

    if (!(batch_size == (*seq_len)->dim_size(0))) {
      return tf::errors::FailedPrecondition(
          "len(sequence_length) != batch_size.  ", "len(sequence_length):  ",
          (*seq_len)->dim_size(0), " batch_size: ", batch_size);
    }

    auto seq_len_t = (*seq_len)->vec<tf::int32>();

    for (int b = 0; b < batch_size; ++b) {
      if (!(seq_len_t(b) <= max_time)) {
        return tf::errors::FailedPrecondition("sequence_length(", b, ") <= ",
                                          max_time);
      }
    }

    tf::Status s = ctx->allocate_output(
        "log_probability", tf::TensorShape({batch_size, top_paths_}), log_prob);
    if (!s.ok()) return s;

    s = ctx->output_list("decoded_indices", decoded_indices);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_values", decoded_values);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_shape", decoded_shape);
    if (!s.ok()) return s;

    return tf::Status::OK();
  }

  // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
  tf::Status StoreAllDecodedSequences(
      const std::vector<std::vector<std::vector<int>>> &sequences,
      tf::OpOutputList *decoded_indices, tf::OpOutputList *decoded_values,
      tf::OpOutputList *decoded_shape) const {
    // Calculate the total number of entries for each path
    const tf::int64 batch_size = sequences.size();
    std::vector<tf::int64> num_entries(top_paths_, 0);

    // Calculate num_entries per path
    for (const auto &batch_s : sequences) {
      CHECK_EQ(batch_s.size(), top_paths_);
      for (int p = 0; p < top_paths_; ++p) {
        num_entries[p] += batch_s[p].size();
      }
    }

    for (int p = 0; p < top_paths_; ++p) {
      tf::Tensor *p_indices = nullptr;
      tf::Tensor *p_values = nullptr;
      tf::Tensor *p_shape = nullptr;

      const tf::int64 p_num = num_entries[p];

      tf::Status s =
          decoded_indices->allocate(p, tf::TensorShape({p_num, 2}), &p_indices);
      if (!s.ok()) return s;
      s = decoded_values->allocate(p, tf::TensorShape({p_num}), &p_values);
      if (!s.ok()) return s;
      s = decoded_shape->allocate(p, tf::TensorShape({2}), &p_shape);
      if (!s.ok()) return s;

      auto indices_t = p_indices->matrix<tf::int64>();
      auto values_t = p_values->vec<tf::int64>();
      auto shape_t = p_shape->vec<tf::int64>();

      tf::int64 max_decoded = 0;
      tf::int64 offset = 0;

      for (tf::int64 b = 0; b < batch_size; ++b) {
        auto &p_batch = sequences[b][p];
        tf::int64 num_decoded = p_batch.size();
        max_decoded = std::max(max_decoded, num_decoded);
        std::copy_n(p_batch.begin(), num_decoded, &values_t(offset));
        for (tf::int64 t = 0; t < num_decoded; ++t, ++offset) {
          indices_t(offset, 0) = b;
          indices_t(offset, 1) = t;
        }
      }

      shape_t(0) = batch_size;
      shape_t(1) = max_decoded;
    }
    return tf::Status::OK();
  }

 private:
  int top_paths_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCDecodeHelper);
};

// CTC beam search
class CTCBeamSearchDecoderOp : public tf::OpKernel {
 public:
  explicit CTCBeamSearchDecoderOp(tf::OpKernelConstruction *ctx)
    : tf::OpKernel(ctx)
    , beam_scorer_(GetModelPath(ctx),
                   GetTriePath(ctx),
                   GetAlphabetPath(ctx))
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
    int top_paths;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));
    decode_helper_.SetTopPaths(top_paths);

    // const tf::Tensor* model_tensor;
    // tf::Status status = ctx->input("model_path", &model_tensor);
    // if (!status.ok()) return status;
    // auto model_vec = model_tensor->flat<std::string>();
    // *model_path = model_vec(0);

    // const tf::Tensor* trie_tensor;
    // status = ctx->input("trie_path", &trie_tensor);
    // if (!status.ok()) return status;
    // auto trie_vec = trie_tensor->flat<std::string>();
    // *trie_path = model_vec(0);

    // const tf::Tensor* alphabet_tensor;
    // status = ctx->input("alphabet_path", &alphabet_tensor);
    // if (!status.ok()) return status;
    // auto alphabet_vec = alphabet_tensor->flat<std::string>();
    // *alphabet_path = alphabet_vec(0);
  }

  std::string GetModelPath(tf::OpKernelConstruction *ctx) {
    std::string model_path;
    ctx->GetAttr("model_path", &model_path);
    return model_path;
  }

  std::string GetTriePath(tf::OpKernelConstruction *ctx) {
    std::string trie_path;
    ctx->GetAttr("trie_path", &trie_path);
    return trie_path;
  }

  std::string GetAlphabetPath(tf::OpKernelConstruction *ctx) {
    std::string alphabet_path;
    ctx->GetAttr("alphabet_path", &alphabet_path);
    return alphabet_path;
  }

  void Compute(tf::OpKernelContext *ctx) override {
    const tf::Tensor *inputs;
    const tf::Tensor *seq_len;
    std::string model_path;
    std::string trie_path;
    std::string alphabet_path;
    tf::Tensor *log_prob = nullptr;
    tf::OpOutputList decoded_indices;
    tf::OpOutputList decoded_values;
    tf::OpOutputList decoded_shape;
    OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
                            ctx, &inputs, &seq_len, &model_path, &trie_path,
                            &alphabet_path, &log_prob, &decoded_indices,
                            &decoded_values, &decoded_shape));

    auto inputs_t = inputs->tensor<float, 3>();
    auto seq_len_t = seq_len->vec<tf::int32>();
    auto log_prob_t = log_prob->matrix<float>();

    const tf::TensorShape &inputs_shape = inputs->shape();

    const tf::int64 max_time = inputs_shape.dim_size(0);
    const tf::int64 batch_size = inputs_shape.dim_size(1);
    const tf::int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        tf::errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    log_prob_t.setZero();

    std::vector<tf::TTypes<float>::UnalignedConstMatrix> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
    }

    tf::ctc::CTCBeamSearchDecoder<KenLMBeamState> beam_search(num_classes, beam_width_,
                                            &beam_scorer_, 1 /* batch_size */,
                                            merge_repeated_);
    tf::Tensor input_chip(tf::DT_FLOAT, tf::TensorShape({num_classes}));
    auto input_chip_t = input_chip.flat<float>();

    std::vector<std::vector<std::vector<int>>> best_paths(batch_size);
    std::vector<float> log_probs;

    // Assumption: the blank index is num_classes - 1
    for (int b = 0; b < batch_size; ++b) {
      auto &best_paths_b = best_paths[b];
      best_paths_b.resize(decode_helper_.GetTopPaths());
      for (int t = 0; t < seq_len_t(b); ++t) {
        input_chip_t = input_list_t[t].chip(b, 0);
        auto input_bi =
            Eigen::Map<const Eigen::ArrayXf>(input_chip_t.data(), num_classes);
        beam_search.Step(input_bi);
      }
      OP_REQUIRES_OK(
          ctx, beam_search.TopPaths(decode_helper_.GetTopPaths(), &best_paths_b,
                                    &log_probs, merge_repeated_));

      beam_search.Reset();

      for (int bp = 0; bp < decode_helper_.GetTopPaths(); ++bp) {
        log_prob_t(b, bp) = log_probs[bp];
      }
    }

    OP_REQUIRES_OK(ctx, decode_helper_.StoreAllDecodedSequences(
                            best_paths, &decoded_indices, &decoded_values,
                            &decoded_shape));
  }

 private:
  CTCDecodeHelper decode_helper_;
  KenLMBeamScorer beam_scorer_;
  bool merge_repeated_;
  int beam_width_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoderOp);
};

REGISTER_KERNEL_BUILDER(Name("CTCBeamSearchDecoderWithLM").Device(tf::DEVICE_CPU),
                        CTCBeamSearchDecoderOp);
