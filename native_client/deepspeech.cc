#ifdef DS_NATIVE_MODEL
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "native_client/deepspeech_model_core.h" // generated
#endif

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "deepspeech.h"
#include "alphabet.h"
#include "beam_search.h"

#include "tensorflow/core/public/version.h"
#include "native_client/ds_version.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include "c_speech_features.h"

//TODO: infer batch size from model/use dynamic batch size
const int BATCH_SIZE = 1;

//TODO: use dynamic sample rate
const int SAMPLE_RATE = 16000;

//TODO: infer n_steps from model
const int N_STEPS_PER_BATCH = 16;

const float AUDIO_WIN_LEN = 0.025f;
const float AUDIO_WIN_STEP = 0.01f;
const int AUDIO_WIN_LEN_SAMPLES = (int)(AUDIO_WIN_LEN * SAMPLE_RATE);
const int AUDIO_WIN_STEP_SAMPLES = (int)(AUDIO_WIN_STEP * SAMPLE_RATE);

const int MFCC_FEATURES = 26;
const int MFCC_CONTEXT = 9;
const int MFCC_WIN_LEN = 2 * MFCC_CONTEXT + 1;
const int MFCC_FEATS_PER_TIMESTEP = MFCC_FEATURES * MFCC_WIN_LEN;

const float PREEMPHASIS_COEFF = 0.97f;
const int N_FFT = 512;
const int N_FILTERS = 26;
const int LOWFREQ = 0;
const int CEP_LIFTER = 22;

using namespace tensorflow;
using tensorflow::ctc::CTCBeamSearchDecoder;
using tensorflow::ctc::CTCDecoder;

using std::vector;

namespace DeepSpeech {

class StreamingState {
public:
  vector<float> accumulated_logits;
  vector<float> audio_buffer;
  float last_sample; // used for preemphasis
  vector<float> mfcc_buffer;
  vector<float> batch_buffer;
  bool skip_next_mfcc;
};

class Private {
public:
  MemmappedEnv* mmap_env;
  Session* session;
  GraphDef graph_def;
  int ncep;
  int ncontext;
  Alphabet* alphabet;
  KenLMBeamScorer* scorer;
  int beam_width;
  bool run_aot;

  Private();
  ~Private();

  /* This is the actual implementation of the streaming inference API, with the
     Model class just forwarding the calls to this class.

     The streaming process uses three buffers that are fed eagerly as audio data
     is fed in. The buffers only hold the minimum amount of data needed to do a
     step in the acoustic model. The three buffers which live in StreamingContext
     are:

     - audio_buffer, used to buffer audio samples until there's enough data to
       compute input features for a single window.

     - mfcc_buffer, used to buffer input features until there's enough data for
       a single timestep. Remember there's overlap in the features, each timestep
       contains N_CONTEXT past feature frames, the current feature frame, and
       N_CONTEXT future feature frames, for a total of MFCC_WIN_LEN feature frames
       per timestep.

     - batch_buffer, used to buffer timesteps until there's enough data to compute
       a batch of N_STEPS_PER_BATCH.

     Data flows through all three buffers as audio samples are fed via the public
     API. When audio_buffer is full, features are computed from it and pushed to
     mfcc_buffer. When mfcc_buffer is full, the timestep is copied to batch_buffer.
     When batch_buffer is full, we do a single step through the acoustic model
     and accumulate results in StreamingState::accumulated_logits.

     When fininshStream() is called, we decode the accumulated logits and return
     the corresponding transcription.
  */
  StreamingState* setupStream(unsigned int prealloc_frames, unsigned int sample_rate);
  void feedAudioContent(StreamingState* ctx, const short* buffer, unsigned int buffer_size);
  const char* finishStream(StreamingState* ctx);

private:
  /**
   * @brief Perform decoding of the logits, using basic CTC decoder or
   *        CTC decoder with KenLM enabled
   *
   * @param n_frames       Number of timesteps to deal with
   * @param logits         Flat matrix of logits, of size:
   *                       n_frames * batch_size * num_classes
   *
   * @return String representing the decoded text.
   */
  const char* decode(vector<float>& logits);

  /**
   * @brief Do a single inference step in the acoustic model, with:
   *          input=mfcc
   *          input_lengths=[n_frames]
   *
   * @param mfcc batch input data
   * @param n_frames number of timesteps in the data
   *
   * @param[out] output_logits Should be large enough to fit
   *                           aNFrames * alphabet_size floats.
   */
  void infer(const float* mfcc, int n_frames, vector<float>& output_logits);

  void processAudioWindow(StreamingState* ctx, const vector<float>& buf);
  void processMfccWindow(StreamingState* ctx, const vector<float>& buf);
  void pushMfccBuffer(StreamingState* ctx, const float* buf, unsigned int len);
  void addZeroMfccWindow(StreamingState* ctx);
  void processBatch(StreamingState* ctx, const vector<float>& buf, unsigned int n_steps);
};

Private::Private()
  : mmap_env(nullptr)
  , session(nullptr)
  , scorer(nullptr)
  , alphabet(nullptr)
  , ncep(0)
  , ncontext(0)
  , beam_width(0)
  , run_aot(false)
{
}

Private::~Private()
{
  if (session) {
    Status status = session->Close();
    if (!status.ok()) {
      std::cerr << "Error closing TensorFlow session: " << status << std::endl;
    }
  }

  delete scorer;
  delete mmap_env;
  delete alphabet;
}

StreamingState*
Private::setupStream(unsigned int prealloc_frames,
                     unsigned int /*sample_rate*/)
{
  Status status = session->Run({}, {}, {"initialize_state"}, nullptr);
  if (!status.ok()) {
    std::cerr << "Error running session: " << status << std::endl;
    return nullptr;
  }

  StreamingState* ctx = new StreamingState;
  if (!ctx) {
    std::cerr << "Could not allocate streaming state." << std::endl;
    return nullptr;
  }

  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

  ctx->accumulated_logits.reserve(prealloc_frames * BATCH_SIZE * num_classes);

  ctx->audio_buffer.reserve(AUDIO_WIN_LEN_SAMPLES);
  ctx->last_sample = 0;
  ctx->mfcc_buffer.reserve(MFCC_FEATS_PER_TIMESTEP);
  ctx->mfcc_buffer.resize(MFCC_FEATURES*MFCC_CONTEXT, 0.f);
  ctx->batch_buffer.reserve(N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP);

  ctx->skip_next_mfcc = false;

  return ctx;
}

void
Private::feedAudioContent(StreamingState* ctx,
                          const short* buffer,
                          unsigned int buffer_size)
{
  // Consume all the data that was passed in, processing full buffers if needed
  while (buffer_size > 0) {
    while (buffer_size > 0 && ctx->audio_buffer.size() < AUDIO_WIN_LEN_SAMPLES) {
      // Apply preemphasis to input sample and buffer it
      float sample = (float)(*buffer) - (PREEMPHASIS_COEFF * ctx->last_sample);
      ctx->audio_buffer.push_back(sample);
      ctx->last_sample = *buffer;
      ++buffer;
      --buffer_size;
    }

    // If the buffer is full, process and shift it
    if (ctx->audio_buffer.size() == AUDIO_WIN_LEN_SAMPLES) {
      processAudioWindow(ctx, ctx->audio_buffer);
      // Shift data by one step of 10ms
      std::rotate(ctx->audio_buffer.begin(), ctx->audio_buffer.begin() + AUDIO_WIN_STEP_SAMPLES, ctx->audio_buffer.end());
      ctx->audio_buffer.resize(ctx->audio_buffer.size() - AUDIO_WIN_STEP_SAMPLES);
    }

    // Repeat until buffer empty
  }
}

const char*
Private::finishStream(StreamingState* ctx)
{
  // Flush audio buffer
  processAudioWindow(ctx, ctx->audio_buffer);

  // Add empty mfcc vectors at end of sample
  for (int i = 0; i < MFCC_CONTEXT; ++i) {
    addZeroMfccWindow(ctx);
  }

  // Process final batch
  if (ctx->batch_buffer.size() > 0) {
    processBatch(ctx, ctx->batch_buffer, ctx->batch_buffer.size()/MFCC_FEATS_PER_TIMESTEP);
  }

  const char* str = decode(ctx->accumulated_logits);
  delete ctx;
  return str;
}

void
Private::processAudioWindow(StreamingState* ctx, const vector<float>& buf)
{
  ctx->skip_next_mfcc = !ctx->skip_next_mfcc;
  if (!ctx->skip_next_mfcc) { // Was true
    return;
  }

  // Compute MFCC features
  float* mfcc;
  int n_frames = csf_mfcc(buf.data(), buf.size(), SAMPLE_RATE,
                          AUDIO_WIN_LEN, AUDIO_WIN_STEP, MFCC_FEATURES, N_FILTERS, N_FFT,
                          LOWFREQ, SAMPLE_RATE/2, 0.f, CEP_LIFTER, 1, nullptr,
                          &mfcc);
  assert(n_frames == 1);

  pushMfccBuffer(ctx, mfcc, n_frames * MFCC_FEATURES);
  free(mfcc);
}

void
Private::addZeroMfccWindow(StreamingState* ctx)
{
  static const float zero_buffer[MFCC_FEATURES] = {0.f};
  pushMfccBuffer(ctx, zero_buffer, MFCC_FEATURES);
}

void
Private::pushMfccBuffer(StreamingState* ctx, const float* buf, unsigned int len)
{
  while (len > 0) {
    unsigned int next_copy_amount = std::min(len, (unsigned int)(MFCC_FEATS_PER_TIMESTEP - ctx->mfcc_buffer.size()));
    ctx->mfcc_buffer.insert(ctx->mfcc_buffer.end(), buf, buf + next_copy_amount);
    buf += next_copy_amount;
    len -= next_copy_amount;
    assert(ctx->mfcc_buffer.size() <= MFCC_FEATS_PER_TIMESTEP);

    if (ctx->mfcc_buffer.size() == MFCC_FEATS_PER_TIMESTEP) {
      processMfccWindow(ctx, ctx->mfcc_buffer);
      // Shift data by one step of one mfcc feature vector
      std::rotate(ctx->mfcc_buffer.begin(), ctx->mfcc_buffer.begin() + MFCC_FEATURES, ctx->mfcc_buffer.end());
      ctx->mfcc_buffer.resize(ctx->mfcc_buffer.size() - MFCC_FEATURES);
    }
  }
}

void
Private::processMfccWindow(StreamingState* ctx, const vector<float>& buf)
{
  auto start = buf.begin();
  while (start != buf.end()) {
    unsigned int next_copy_amount = std::min(std::distance(start, buf.end()), (long)(N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP - ctx->batch_buffer.size()));
    ctx->batch_buffer.insert(ctx->batch_buffer.end(), start, start + next_copy_amount);
    start += next_copy_amount;
    assert(ctx->batch_buffer.size() <= N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP);

    if (ctx->batch_buffer.size() == N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP) {
      processBatch(ctx, ctx->batch_buffer, N_STEPS_PER_BATCH);
      ctx->batch_buffer.resize(0);
    }
  }
}

void
Private::processBatch(StreamingState* ctx, const vector<float>& buf, unsigned int n_steps)
{
  infer(buf.data(), n_steps, ctx->accumulated_logits);
}

void
Private::infer(const float* aMfcc, int n_frames, vector<float>& logits_output)
{
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

  if (run_aot) {
#ifdef DS_NATIVE_MODEL
    Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

    nativeModel nm(nativeModel::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);
    nm.set_thread_pool(&device);

    for (int ot = 0; ot < n_frames; ot += DS_MODEL_TIMESTEPS) {
      nm.set_arg0_data(&(aMfcc[ot * MFCC_FEATS_PER_TIMESTEP]));
      nm.Run();

      // The CTCDecoder works with log-probs.
      for (int t = 0; t < DS_MODEL_TIMESTEPS, (ot + t) < n_frames; ++t) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
          for (int c = 0; c < num_classes; ++c) {
            logits_output.push_back(nm.result0(t, b, c));
          }
        }
      }
    }
#else
    std::cerr << "No support for native model built-in." << std::endl;
    return;
#endif // DS_NATIVE_MODEL
  } else {
    Tensor input(DT_FLOAT, TensorShape({BATCH_SIZE, N_STEPS_PER_BATCH, MFCC_FEATS_PER_TIMESTEP}));

    auto input_mapped = input.tensor<float, 3>();
    int idx = 0;
    for (int i = 0; i < n_frames; i++) {
      for (int j = 0; j < MFCC_FEATS_PER_TIMESTEP; j++, idx++) {
        input_mapped(0, i, j) = aMfcc[idx];
      }
    }

    Tensor input_lengths(DT_INT32, TensorShape({1}));
    input_lengths.scalar<int>()() = n_frames;

    vector<Tensor> outputs;
    Status status = session->Run(
      {{"input_node", input}, {"input_lengths", input_lengths}},
      {"logits"}, {}, &outputs);

    if (!status.ok()) {
      std::cerr << "Error running session: " << status << "\n";
      return;
    }

    auto logits_mapped = outputs[0].flat<float>();
    // The CTCDecoder works with log-probs.
    for (int t = 0; t < n_frames * BATCH_SIZE * num_classes; ++t) {
      logits_output.push_back(logits_mapped(t));
    }
  }
}

const char*
Private::decode(vector<float>& logits)
{
  const int top_paths = 1;
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank
  const int n_frames = logits.size() / (BATCH_SIZE * num_classes);

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[BATCH_SIZE] = {n_frames};

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], BATCH_SIZE);
  vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(n_frames);
  for (int t = 0; t < n_frames; ++t) {
    inputs.emplace_back(&logits[t * BATCH_SIZE * num_classes], BATCH_SIZE, num_classes);
  }

  // Prepare containers for output and scores.
  // CTCDecoder::Output is vector<vector<int>>
  vector<CTCDecoder::Output> decoder_outputs(top_paths);
  for (CTCDecoder::Output& output : decoder_outputs) {
    output.resize(BATCH_SIZE);
  }
  float score[BATCH_SIZE][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], BATCH_SIZE, top_paths);

  if (scorer == nullptr) {
    CTCBeamSearchDecoder<>::DefaultBeamScorer default_scorer;
    CTCBeamSearchDecoder<> decoder(num_classes,
                                   beam_width,
                                   &default_scorer,
                                   BATCH_SIZE);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  } else {
    CTCBeamSearchDecoder<KenLMBeamState> decoder(num_classes,
                                                 beam_width,
                                                 scorer,
                                                 BATCH_SIZE);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  }

  // Output is an array of shape (batch_size, top_paths, result_length).

  std::stringstream output;
  for (int64 character : decoder_outputs[0][0]) {
    output << alphabet->StringFromLabel(character);
  }

  return strdup(output.str().c_str());
}

DEEPSPEECH_EXPORT
Model::Model(const char* aModelPath, int aNCep, int aNContext,
             const char* aAlphabetConfigPath, int aBeamWidth)
{
  mPriv             = new Private();
  mPriv->mmap_env   = new MemmappedEnv(Env::Default());
  mPriv->ncep       = aNCep;
  mPriv->ncontext   = aNContext;
  mPriv->alphabet   = new Alphabet(aAlphabetConfigPath);
  mPriv->beam_width = aBeamWidth;
  mPriv->run_aot    = false;

  print_versions();

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, will rely on built-in model." << std::endl;
    mPriv->run_aot = true;
    return;
  }

  Status status;
  SessionOptions options;

  bool is_mmap = std::string(aModelPath).find(".pbmm") != std::string::npos;
  if (!is_mmap) {
    std::cerr << "Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage." << std::endl;
  } else {
    status = mPriv->mmap_env->InitializeFromFile(aModelPath);
    if (!status.ok()) {
      std::cerr << status << std::endl;
      return;
    }

    options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(::OptimizerOptions::L0);
    options.env = mPriv->mmap_env;
  }

  status = NewSession(options, &mPriv->session);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return;
  }

  if (is_mmap) {
    status = ReadBinaryProto(mPriv->mmap_env,
                             MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                             &mPriv->graph_def);
  } else {
    status = ReadBinaryProto(Env::Default(), aModelPath, &mPriv->graph_def);
  }
  if (!status.ok()) {
    std::cerr << status << std::endl;
    delete mPriv;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    delete mPriv;
    return;
  }

  for (int i = 0; i < mPriv->graph_def.node_size(); ++i) {
    NodeDef node = mPriv->graph_def.node(i);
    if (node.name() == "logits_shape") {
      Tensor logits_shape = Tensor(DT_INT32, TensorShape({3}));
      if (!logits_shape.FromProto(node.attr().at("value").tensor())) {
        break;
      }

      int final_dim_size = logits_shape.vec<int>()(2) - 1;
      if (final_dim_size != mPriv->alphabet->GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << mPriv->alphabet->GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        delete mPriv;
        return;
      }
      break;
    }
  }
}

DEEPSPEECH_EXPORT
Model::~Model()
{
  delete mPriv;
}

DEEPSPEECH_EXPORT
void
Model::enableDecoderWithLM(const char* aAlphabetConfigPath, const char* aLMPath,
                           const char* aTriePath, float aLMWeight,
                           float aWordCountWeight, float aValidWordCountWeight)
{
  mPriv->scorer = new KenLMBeamScorer(aLMPath, aTriePath, aAlphabetConfigPath,
                                      aLMWeight, aWordCountWeight, aValidWordCountWeight);
}

DEEPSPEECH_EXPORT
void
Model::getInputVector(const short* aBuffer, unsigned int aBufferSize,
                      int aSampleRate, float** aMfcc, int* aNFrames,
                      int* aFrameLen)
{
  return audioToInputVector(aBuffer, aBufferSize, aSampleRate, mPriv->ncep,
                            mPriv->ncontext, aMfcc, aNFrames, aFrameLen);
}

DEEPSPEECH_EXPORT
const char*
Model::stt(const short* aBuffer,
           unsigned int aBufferSize,
           int aSampleRate)
{
  StreamingState* ctx = setupStream();
  if (!ctx) {
    return nullptr;
  }
  feedAudioContent(ctx, aBuffer, aBufferSize);
  return finishStream(ctx);
}

DEEPSPEECH_EXPORT
StreamingState*
Model::setupStream(unsigned int aPreAllocFrames,
                   unsigned int aSampleRate)
{
  return mPriv->setupStream(aPreAllocFrames, aSampleRate);
}

DEEPSPEECH_EXPORT
void
Model::feedAudioContent(StreamingState* ctx,
                        const short* aBuffer,
                        unsigned int aBufferSize)
{
  mPriv->feedAudioContent(ctx, aBuffer, aBufferSize);
}

DEEPSPEECH_EXPORT
const char*
Model::finishStream(StreamingState* ctx)
{
  return mPriv->finishStream(ctx);
}

DEEPSPEECH_EXPORT
void
audioToInputVector(const short* aBuffer, unsigned int aBufferSize,
                   int aSampleRate, int aNCep, int aNContext, float** aMfcc,
                   int* aNFrames, int* aFrameLen)
{
  const int contextSize = aNCep * aNContext;
  const int frameSize = aNCep + (2 * aNCep * aNContext);

  // Compute MFCC features
  float* mfcc;
  int n_frames = csf_mfcc(aBuffer, aBufferSize, aSampleRate,
                          AUDIO_WIN_LEN, AUDIO_WIN_STEP, aNCep, N_FILTERS, N_FFT,
                          LOWFREQ, aSampleRate/2, PREEMPHASIS_COEFF, CEP_LIFTER,
                          1, NULL, &mfcc);

  // Take every other frame (BiRNN stride of 2) and add past/future context
  int ds_input_length = (n_frames + 1) / 2;
  // TODO: Use MFCC of silence instead of zero
  float* ds_input = (float*)calloc(ds_input_length * frameSize, sizeof(float));
  for (int i = 0, idx = 0, mfcc_idx = 0; i < ds_input_length;
       i++, idx += frameSize, mfcc_idx += aNCep * 2) {
    // Past context
    for (int j = aNContext; j > 0; j--) {
      int frame_index = (i - j) * 2;
      if (frame_index < 0) { continue; }
      int mfcc_base = frame_index * aNCep;
      int base = (aNContext - j) * aNCep;
      for (int k = 0; k < aNCep; k++) {
        ds_input[idx + base + k] = mfcc[mfcc_base + k];
      }
    }

    // Present context
    for (int j = 0; j < aNCep; j++) {
      ds_input[idx + j + contextSize] = mfcc[mfcc_idx + j];
    }

    // Future context
    for (int j = 1; j <= aNContext; j++) {
      int frame_index = (i + j) * 2;
      if (frame_index >= n_frames) { break; }
      int mfcc_base = frame_index * aNCep;
      int base = contextSize + aNCep + ((j - 1) * aNCep);
      for (int k = 0; k < aNCep; k++) {
        ds_input[idx + base + k] = mfcc[mfcc_base + k];
      }
    }
  }

  // Free mfcc array
  free(mfcc);

  if (aMfcc) {
    *aMfcc = ds_input;
  }
  if (aNFrames) {
    *aNFrames = ds_input_length;
  }
  if (aFrameLen) {
    *aFrameLen = frameSize;
  }
}

DEEPSPEECH_EXPORT
void
print_versions() {
  std::cerr << "TensorFlow: " << tf_git_version() << std::endl;
  std::cerr << "DeepSpeech: " << ds_git_version() << std::endl;
}

}
