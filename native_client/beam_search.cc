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

#include <algorithm>
#include <vector>
#include <cmath>

#include "beam_search.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"

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
    .Attr("lm_weight: float")
    .Attr("word_count_weight: float")
    .Attr("valid_word_count_weight: float")
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
model_path: A string containing the path to the KenLM model file to use.
trie_path: A string containing the path to the trie file built from the vocabulary.
alphabet_path: A string containing the path to the alphabet file (see alphabet.h).
lm_weight: alpha hyperparameter of CTC decoder. LM weight.
word_count_weight: beta hyperparameter of CTC decoder. Word insertion weight.
valid_word_count_weight: beta' hyperparameter of CTC decoder. Valid word insertion weight.
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

class CTCBeamSearchDecoderWithLMOp : public tf::OpKernel {
 public:
  explicit CTCBeamSearchDecoderWithLMOp(tf::OpKernelConstruction *ctx)
    : tf::OpKernel(ctx)
    , beam_scorer_(GetModelPath(ctx),
                   GetTriePath(ctx),
                   GetAlphabetPath(ctx),
                   GetLMWeight(ctx),
                   GetWordCountWeight(ctx),
                   GetValidWordCountWeight(ctx))
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
    int top_paths;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));
    decode_helper_.SetTopPaths(top_paths);
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
  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoderWithLMOp);

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

  float GetLMWeight(tf::OpKernelConstruction *ctx) {
    float lm_weight;
    ctx->GetAttr("lm_weight", &lm_weight);
    return lm_weight;
  }

  float GetWordCountWeight(tf::OpKernelConstruction *ctx) {
    float word_count_weight;
    ctx->GetAttr("word_count_weight", &word_count_weight);
    return word_count_weight;
  }

  float GetValidWordCountWeight(tf::OpKernelConstruction *ctx) {
    float valid_word_count_weight;
    ctx->GetAttr("valid_word_count_weight", &valid_word_count_weight);
    return valid_word_count_weight;
  }
};

REGISTER_KERNEL_BUILDER(Name("CTCBeamSearchDecoderWithLM").Device(tf::DEVICE_CPU),
                        CTCBeamSearchDecoderWithLMOp);
