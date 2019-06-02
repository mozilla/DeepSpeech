#ifndef TFMODELSTATE_H
#define TFMODELSTATE_H

#include <vector>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include "modelstate.h"

struct TFModelState : public ModelState
{
  tensorflow::MemmappedEnv* mmap_env_;
  tensorflow::Session* session_;
  tensorflow::GraphDef graph_def_;

  TFModelState();
  virtual ~TFModelState();

  virtual int init(const char* model_path,
                   unsigned int n_features,
                   unsigned int n_context,
                   const char* alphabet_path,
                   unsigned int beam_width) override;

  virtual int initialize_state() override;

  virtual void infer(const float* mfcc,
                     unsigned int n_frames,
                     std::vector<float>& logits_output) override;

  virtual void compute_mfcc(const std::vector<float>& audio_buffer,
                            std::vector<float>& mfcc_output) override;
};

#endif // TFMODELSTATE_H
