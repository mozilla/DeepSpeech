#include <algorithm>
#ifdef _MSC_VER
  #define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepspeech.h"
#include "alphabet.h"
#include "modelstate.h"

#include "workspace_status.h"

#ifndef USE_TFLITE
#include "tfmodelstate.h"
#else
#include "tflitemodelstate.h"
#endif // USE_TFLITE

#include "ctcdecode/ctc_beam_search_decoder.h"

#ifdef __ANDROID__
#include <android/log.h>
#define  LOG_TAG    "libdeepspeech"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define  LOGD(...)
#define  LOGE(...)
#endif // __ANDROID__

using std::vector;

/* This is the implementation of the streaming inference API.

   The streaming process uses three buffers that are fed eagerly as audio data
   is fed in. The buffers only hold the minimum amount of data needed to do a
   step in the acoustic model. The three buffers which live in StreamingState
   are:

   - audio_buffer, used to buffer audio samples until there's enough data to
     compute input features for a single window.

   - mfcc_buffer, used to buffer input features until there's enough data for
     a single timestep. Remember there's overlap in the features, each timestep
     contains n_context past feature frames, the current feature frame, and
     n_context future feature frames, for a total of 2*n_context + 1 feature
     frames per timestep.

   - batch_buffer, used to buffer timesteps until there's enough data to compute
     a batch of n_steps.

   Data flows through all three buffers as audio samples are fed via the public
   API. When audio_buffer is full, features are computed from it and pushed to
   mfcc_buffer. When mfcc_buffer is full, the timestep is copied to batch_buffer.
   When batch_buffer is full, we do a single step through the acoustic model
   and accumulate the intermediate decoding state in the DecoderState structure.

   When finishStream() is called, we return the corresponding transcript from
   the current decoder state.
*/
struct StreamingState {
  vector<float> audio_buffer_;
  vector<float> mfcc_buffer_;
  vector<float> batch_buffer_;
  vector<float> previous_state_c_;
  vector<float> previous_state_h_;

  ModelState* model_;
  DecoderState decoder_state_;

  StreamingState();
  ~StreamingState();

  void feedAudioContent(const short* buffer, unsigned int buffer_size);
  char* intermediateDecode() const;
  Metadata* intermediateDecodeWithMetadata(unsigned int num_results) const;
  void finalizeStream();
  char* finishStream();
  Metadata* finishStreamWithMetadata(unsigned int num_results);

  void processAudioWindow(const vector<float>& buf);
  void processMfccWindow(const vector<float>& buf);
  void pushMfccBuffer(const vector<float>& buf);
  void addZeroMfccWindow();
  void processBatch(const vector<float>& buf, unsigned int n_steps);
};

StreamingState::StreamingState()
{
}

StreamingState::~StreamingState()
{
}

template<typename T>
void
shift_buffer_left(vector<T>& buf, int shift_amount)
{
  std::rotate(buf.begin(), buf.begin() + shift_amount, buf.end());
  buf.resize(buf.size() - shift_amount);
}

void
StreamingState::feedAudioContent(const short* buffer,
                                 unsigned int buffer_size)
{
  // Consume all the data that was passed in, processing full buffers if needed
  while (buffer_size > 0) {
    while (buffer_size > 0 && audio_buffer_.size() < model_->audio_win_len_) {
      // Convert i16 sample into f32
      float multiplier = 1.0f / (1 << 15);
      audio_buffer_.push_back((float)(*buffer) * multiplier);
      ++buffer;
      --buffer_size;
    }

    // If the buffer is full, process and shift it
    if (audio_buffer_.size() == model_->audio_win_len_) {
      processAudioWindow(audio_buffer_);
      // Shift data by one step
      shift_buffer_left(audio_buffer_, model_->audio_win_step_);
    }

    // Repeat until buffer empty
  }
}

char*
StreamingState::intermediateDecode() const
{
  return model_->decode(decoder_state_);
}

Metadata*
StreamingState::intermediateDecodeWithMetadata(unsigned int num_results) const
{
  return model_->decode_metadata(decoder_state_, num_results);
}

char*
StreamingState::finishStream()
{
  finalizeStream();
  return model_->decode(decoder_state_);
}

Metadata*
StreamingState::finishStreamWithMetadata(unsigned int num_results)
{
  finalizeStream();
  return model_->decode_metadata(decoder_state_, num_results);
}

void
StreamingState::processAudioWindow(const vector<float>& buf)
{
  // Compute MFCC features
  vector<float> mfcc;
  mfcc.reserve(model_->n_features_);
  model_->compute_mfcc(buf, mfcc);
  pushMfccBuffer(mfcc);
}

void
StreamingState::finalizeStream()
{
  // Flush audio buffer
  processAudioWindow(audio_buffer_);

  // Add empty mfcc vectors at end of sample
  for (int i = 0; i < model_->n_context_; ++i) {
    addZeroMfccWindow();
  }

  // Process final batch
  if (batch_buffer_.size() > 0) {
    processBatch(batch_buffer_, batch_buffer_.size()/model_->mfcc_feats_per_timestep_);
  }
}

void
StreamingState::addZeroMfccWindow()
{
  vector<float> zero_buffer(model_->n_features_, 0.f);
  pushMfccBuffer(zero_buffer);
}

template<typename InputIt, typename OutputIt>
InputIt
copy_up_to_n(InputIt from_begin, InputIt from_end, OutputIt to_begin, int max_elems)
{
  int next_copy_amount = std::min<int>(std::distance(from_begin, from_end), max_elems);
  std::copy_n(from_begin, next_copy_amount, to_begin);
  return from_begin + next_copy_amount;
}

void
StreamingState::pushMfccBuffer(const vector<float>& buf)
{
  auto start = buf.begin();
  auto end = buf.end();
  while (start != end) {
    // Copy from input buffer to mfcc_buffer, stopping if we have a full context window
    start = copy_up_to_n(start, end, std::back_inserter(mfcc_buffer_),
                         model_->mfcc_feats_per_timestep_ - mfcc_buffer_.size());
    assert(mfcc_buffer_.size() <= model_->mfcc_feats_per_timestep_);

    // If we have a full context window
    if (mfcc_buffer_.size() == model_->mfcc_feats_per_timestep_) {
      processMfccWindow(mfcc_buffer_);
      // Shift data by one step of one mfcc feature vector
      shift_buffer_left(mfcc_buffer_, model_->n_features_);
    }
  }
}

void
StreamingState::processMfccWindow(const vector<float>& buf)
{
  auto start = buf.begin();
  auto end = buf.end();
  while (start != end) {
    // Copy from input buffer to batch_buffer, stopping if we have a full batch
    start = copy_up_to_n(start, end, std::back_inserter(batch_buffer_),
                         model_->n_steps_ * model_->mfcc_feats_per_timestep_ - batch_buffer_.size());
    assert(batch_buffer_.size() <= model_->n_steps_ * model_->mfcc_feats_per_timestep_);

    // If we have a full batch
    if (batch_buffer_.size() == model_->n_steps_ * model_->mfcc_feats_per_timestep_) {
      processBatch(batch_buffer_, model_->n_steps_);
      batch_buffer_.resize(0);
    }
  }
}

void
StreamingState::processBatch(const vector<float>& buf, unsigned int n_steps)
{
  vector<float> logits;
  model_->infer(buf,
                n_steps,
                previous_state_c_,
                previous_state_h_,
                logits,
                previous_state_c_,
                previous_state_h_);

  const size_t num_classes = model_->alphabet_.GetSize() + 1; // +1 for blank
  const int n_frames = logits.size() / (ModelState::BATCH_SIZE * num_classes);

  // Convert logits to double
  vector<double> inputs(logits.begin(), logits.end());

  decoder_state_.next(inputs.data(),
                      n_frames,
                      num_classes);
}

int
DS_CreateModel(const char* aModelPath,
               ModelState** retval)
{
  *retval = nullptr;

  std::cerr << "TensorFlow: " << tf_local_git_version() << std::endl;
  std::cerr << "DeepSpeech: " << ds_git_version() << std::endl;
#ifdef __ANDROID__
  LOGE("TensorFlow: %s", tf_local_git_version());
  LOGD("TensorFlow: %s", tf_local_git_version());
  LOGE("DeepSpeech: %s", ds_git_version());
  LOGD("DeepSpeech: %s", ds_git_version());
#endif

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, cannot continue." << std::endl;
    return DS_ERR_NO_MODEL;
  }

  std::unique_ptr<ModelState> model(
#ifndef USE_TFLITE
    new TFModelState()
#else
    new TFLiteModelState()
#endif
  );

  if (!model) {
    std::cerr << "Could not allocate model state." << std::endl;
    return DS_ERR_FAIL_CREATE_MODEL;
  }

  int err = model->init(aModelPath);
  if (err != DS_ERR_OK) {
    return err;
  }

  *retval = model.release();
  return DS_ERR_OK;
}

unsigned int
DS_GetModelBeamWidth(const ModelState* aCtx)
{
  return aCtx->beam_width_;
}

int
DS_SetModelBeamWidth(ModelState* aCtx, unsigned int aBeamWidth)
{
  aCtx->beam_width_ = aBeamWidth;
  return 0;
}

int
DS_GetModelSampleRate(const ModelState* aCtx)
{
  return aCtx->sample_rate_;
}

void
DS_FreeModel(ModelState* ctx)
{
  delete ctx;
}

int
DS_EnableExternalScorer(ModelState* aCtx,
                        const char* aScorerPath)
{
  std::unique_ptr<Scorer> scorer(new Scorer());
  int err = scorer->init(aScorerPath, aCtx->alphabet_);
  if (err != 0) {
    return DS_ERR_INVALID_SCORER;
  }
  aCtx->scorer_ = std::move(scorer);
  return DS_ERR_OK;
}

int
DS_DisableExternalScorer(ModelState* aCtx)
{
  if (aCtx->scorer_) {
    aCtx->scorer_.reset();
    return DS_ERR_OK;
  }
  return DS_ERR_SCORER_NOT_ENABLED;
}

int DS_SetScorerAlphaBeta(ModelState* aCtx,
                          float aAlpha,
                          float aBeta)
{
  if (aCtx->scorer_) {
    aCtx->scorer_->reset_params(aAlpha, aBeta);
    return DS_ERR_OK;
  }
  return DS_ERR_SCORER_NOT_ENABLED;
}

int
DS_CreateStream(ModelState* aCtx,
                StreamingState** retval)
{
  *retval = nullptr;

  std::unique_ptr<StreamingState> ctx(new StreamingState());
  if (!ctx) {
    std::cerr << "Could not allocate streaming state." << std::endl;
    return DS_ERR_FAIL_CREATE_STREAM;
  }

  ctx->audio_buffer_.reserve(aCtx->audio_win_len_);
  ctx->mfcc_buffer_.reserve(aCtx->mfcc_feats_per_timestep_);
  ctx->mfcc_buffer_.resize(aCtx->n_features_*aCtx->n_context_, 0.f);
  ctx->batch_buffer_.reserve(aCtx->n_steps_ * aCtx->mfcc_feats_per_timestep_);
  ctx->previous_state_c_.resize(aCtx->state_size_, 0.f);
  ctx->previous_state_h_.resize(aCtx->state_size_, 0.f);
  ctx->model_ = aCtx;

  const int cutoff_top_n = 40;
  const double cutoff_prob = 1.0;

  ctx->decoder_state_.init(aCtx->alphabet_,
                           aCtx->beam_width_,
                           cutoff_prob,
                           cutoff_top_n,
                           aCtx->scorer_);

  *retval = ctx.release();
  return DS_ERR_OK;
}

void
DS_FeedAudioContent(StreamingState* aSctx,
                    const short* aBuffer,
                    unsigned int aBufferSize)
{
  aSctx->feedAudioContent(aBuffer, aBufferSize);
}

char*
DS_IntermediateDecode(const StreamingState* aSctx)
{
  return aSctx->intermediateDecode();
}

Metadata*
DS_IntermediateDecodeWithMetadata(const StreamingState* aSctx,
                                  unsigned int aNumResults)
{
  return aSctx->intermediateDecodeWithMetadata(aNumResults);
}

char*
DS_FinishStream(StreamingState* aSctx)
{
  char* str = aSctx->finishStream();
  DS_FreeStream(aSctx);
  return str;
}

Metadata*
DS_FinishStreamWithMetadata(StreamingState* aSctx, 
                            unsigned int aNumResults)
{
  Metadata* result = aSctx->finishStreamWithMetadata(aNumResults);
  DS_FreeStream(aSctx);
  return result;
}

StreamingState*
CreateStreamAndFeedAudioContent(ModelState* aCtx,
                                const short* aBuffer,
                                unsigned int aBufferSize)
{
  StreamingState* ctx;
  int status = DS_CreateStream(aCtx, &ctx);
  if (status != DS_ERR_OK) {
    return nullptr;
  }
  DS_FeedAudioContent(ctx, aBuffer, aBufferSize);
  return ctx;
}

char*
DS_SpeechToText(ModelState* aCtx,
                const short* aBuffer,
                unsigned int aBufferSize)
{
  StreamingState* ctx = CreateStreamAndFeedAudioContent(aCtx, aBuffer, aBufferSize);
  return DS_FinishStream(ctx);
}

Metadata*
DS_SpeechToTextWithMetadata(ModelState* aCtx,
                            const short* aBuffer,
                            unsigned int aBufferSize,
                            unsigned int aNumResults)
{
  StreamingState* ctx = CreateStreamAndFeedAudioContent(aCtx, aBuffer, aBufferSize);
  return DS_FinishStreamWithMetadata(ctx, aNumResults);
}

void
DS_FreeStream(StreamingState* aSctx)
{
  delete aSctx;
}

void
DS_FreeMetadata(Metadata* m)
{
  if (m) {
    for (int i = 0; i < m->num_transcripts; ++i) {
      for (int j = 0; j < m->transcripts[i].num_tokens; ++j) {
        free((void*)m->transcripts[i].tokens[j].text);
      }

      free((void*)m->transcripts[i].tokens);
    }

    free((void*)m->transcripts);
    free(m);
  }
}

void
DS_FreeString(char* str)
{
  free(str);
}

char*
DS_Version()
{
  return strdup(ds_version());
}
