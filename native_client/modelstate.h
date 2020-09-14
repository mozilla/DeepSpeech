#ifndef MODELSTATE_H
#define MODELSTATE_H

#include <vector>

#include "deepspeech.h"
#include "alphabet.h"

#include "ctcdecode/scorer.h"
#include "ctcdecode/output.h"

class DecoderState;

struct ModelState {
  //TODO: infer batch size from model/use dynamic batch size
  static constexpr unsigned int BATCH_SIZE = 1;

  Alphabet alphabet_;
  std::shared_ptr<Scorer> scorer_;
  std::unordered_map<std::string, float> hot_words_;
  unsigned int beam_width_;
  unsigned int n_steps_;
  unsigned int n_context_;
  unsigned int n_features_;
  unsigned int mfcc_feats_per_timestep_;
  unsigned int sample_rate_;
  unsigned int audio_win_len_;
  unsigned int audio_win_step_;
  unsigned int state_size_;

  ModelState();
  virtual ~ModelState();

  virtual int init(const char* model_path);

  virtual void compute_mfcc(const std::vector<float>& audio_buffer, std::vector<float>& mfcc_output) = 0;

  /**
   * @brief Do a single inference step in the acoustic model, with:
   *          input=mfcc
   *          input_lengths=[n_frames]
   *
   * @param mfcc batch input data
   * @param n_frames number of timesteps in the data
   *
   * @param[out] output_logits Where to store computed logits.
   */
  virtual void infer(const std::vector<float>& mfcc,
                     unsigned int n_frames,
                     const std::vector<float>& previous_state_c,
                     const std::vector<float>& previous_state_h,
                     std::vector<float>& logits_output,
                     std::vector<float>& state_c_output,
                     std::vector<float>& state_h_output) = 0;

  /**
   * @brief Perform decoding of the logits, using basic CTC decoder or
   *        CTC decoder with KenLM enabled
   *
   * @param state Decoder state to use when decoding.
   *
   * @return String representing the decoded text.
   */
  virtual char* decode(const DecoderState& state) const;

  /**
   * @brief Return character-level metadata including letter timings.
   *
   * @param state Decoder state to use when decoding.
   * @param num_results Maximum number of candidate results to return.
   *
   * @return A Metadata struct containing CandidateTranscript structs.
   * Each represents an candidate transcript, with the first ranked most probable.
   * The user is responsible for freeing Result by calling DS_FreeMetadata().
   */
  virtual Metadata* decode_metadata(const DecoderState& state,
                                    size_t num_results);
};

#endif // MODELSTATE_H
