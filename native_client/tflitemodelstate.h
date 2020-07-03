#ifndef TFLITEMODELSTATE_H
#define TFLITEMODELSTATE_H

#include <memory>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include "modelstate.h"

struct TFLiteModelState : public ModelState
{
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> fbmodel_;

  int input_node_idx_;
  int previous_state_c_idx_;
  int previous_state_h_idx_;
  int input_samples_idx_;

  int logits_idx_;
  int new_state_c_idx_;
  int new_state_h_idx_;
  int mfccs_idx_;

  std::vector<int> acoustic_exec_plan_;
  std::vector<int> mfcc_exec_plan_;

  TFLiteModelState();
  virtual ~TFLiteModelState();

  virtual int init(const char* model_path) override;

  virtual void compute_mfcc(const std::vector<float>& audio_buffer,
                            std::vector<float>& mfcc_output) override;

  virtual void infer(const std::vector<float>& mfcc,
                     unsigned int n_frames,
                     const std::vector<float>& previous_state_c,
                     const std::vector<float>& previous_state_h,
                     std::vector<float>& logits_output,
                     std::vector<float>& state_c_output,
                     std::vector<float>& state_h_output) override;

private:
  int get_tensor_by_name(const std::vector<int>& list, const char* name);
  int get_input_tensor_by_name(const char* name);
  int get_output_tensor_by_name(const char* name);
  std::vector<int> find_parent_node_ids(int tensor_id);
  void copy_vector_to_tensor(const std::vector<float>& vec,
                             int tensor_idx,
                             int num_elements);
  void copy_tensor_to_vector(int tensor_idx,
                             int num_elements,
                             std::vector<float>& vec);
};

#endif // TFLITEMODELSTATE_H
