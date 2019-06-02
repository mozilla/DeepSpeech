#ifndef TFLITEMODELSTATE_H
#define TFLITEMODELSTATE_H

#include <memory>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

#include "modelstate.h"

struct TFLiteModelState : public ModelState
{
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> fbmodel_;

  size_t previous_state_size_;
  std::unique_ptr<float[]> previous_state_c_;
  std::unique_ptr<float[]> previous_state_h_;

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

  virtual int init(const char* model_path,
                   unsigned int n_features,
                   unsigned int n_context,
                   const char* alphabet_path,
                   unsigned int beam_width) override;

  virtual int initialize_state() override;
  
  virtual void compute_mfcc(const std::vector<float>& audio_buffer,
                            std::vector<float>& mfcc_output) override;

  virtual void infer(const float* mfcc, unsigned int n_frames,
                     std::vector<float>& logits_output) override;
};

#endif // TFLITEMODELSTATE_H
