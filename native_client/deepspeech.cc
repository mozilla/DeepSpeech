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

#include "native_client/ds_version.h"
#include "native_client/ds_graph_version.h"

#ifndef USE_TFLITE
  #include "tensorflow/core/public/session.h"
  #include "tensorflow/core/platform/env.h"
  #include "tensorflow/core/util/memmapped_file_system.h"
#else // USE_TFLITE
  #include "tensorflow/lite/model.h"
  #include "tensorflow/lite/kernels/register.h"
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

//TODO: infer batch size from model/use dynamic batch size
constexpr unsigned int BATCH_SIZE = 1;

constexpr unsigned int DEFAULT_SAMPLE_RATE = 16000;
constexpr unsigned int DEFAULT_WINDOW_LENGTH = DEFAULT_SAMPLE_RATE * 0.032;
constexpr unsigned int DEFAULT_WINDOW_STEP = DEFAULT_SAMPLE_RATE * 0.02;

#ifndef USE_TFLITE
  using namespace tensorflow;
#else
  using namespace tflite;
#endif

using std::vector;

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
     contains n_context past feature frames, the current feature frame, and
     n_context future feature frames, for a total of 2*n_context + 1 feature
     frames per timestep.

   - batch_buffer, used to buffer timesteps until there's enough data to compute
     a batch of n_steps.

   Data flows through all three buffers as audio samples are fed via the public
   API. When audio_buffer is full, features are computed from it and pushed to
   mfcc_buffer. When mfcc_buffer is full, the timestep is copied to batch_buffer.
   When batch_buffer is full, we do a single step through the acoustic model
   and accumulate results in StreamingState::accumulated_logits.

   When fininshStream() is called, we decode the accumulated logits and return
   the corresponding transcription.
*/
struct StreamingState {
  vector<float> accumulated_logits;
  vector<float> audio_buffer;
  vector<float> mfcc_buffer;
  vector<float> batch_buffer;
  ModelState* model;

  void feedAudioContent(const short* buffer, unsigned int buffer_size);
  char* intermediateDecode();
  void finalizeStream();
  char* finishStream();
  Metadata* finishStreamWithMetadata();

  void processAudioWindow(const vector<float>& buf);
  void processMfccWindow(const vector<float>& buf);
  void pushMfccBuffer(const vector<float>& buf);
  void addZeroMfccWindow();
  void processBatch(const vector<float>& buf, unsigned int n_steps);
};

struct ModelState {
#ifndef USE_TFLITE
  MemmappedEnv* mmap_env;
  Session* session;
  GraphDef graph_def;
#else // USE_TFLITE
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<FlatBufferModel> fbmodel;
#endif // USE_TFLITE
  unsigned int ncep;
  unsigned int ncontext;
  Alphabet* alphabet;
  Scorer* scorer;
  unsigned int beam_width;
  unsigned int n_steps;
  unsigned int n_context;
  unsigned int n_features;
  unsigned int mfcc_feats_per_timestep;
  unsigned int sample_rate;
  unsigned int audio_win_len;
  unsigned int audio_win_step;

#ifdef USE_TFLITE
  size_t previous_state_size;
  std::unique_ptr<float[]> previous_state_c_;
  std::unique_ptr<float[]> previous_state_h_;

  int input_node_idx;
  int previous_state_c_idx;
  int previous_state_h_idx;
  int input_samples_idx;

  int logits_idx;
  int new_state_c_idx;
  int new_state_h_idx;
  int mfccs_idx;
#endif

  ModelState();
  ~ModelState();

  /**
   * @brief Perform decoding of the logits, using basic CTC decoder or
   *        CTC decoder with KenLM enabled
   *
   * @param logits         Flat matrix of logits, of size:
   *                       n_frames * batch_size * num_classes
   *
   * @return String representing the decoded text.
   */
  char* decode(const vector<float>& logits);

  /**
   * @brief Perform decoding of the logits, using basic CTC decoder or
   *        CTC decoder with KenLM enabled
   *
   * @param logits         Flat matrix of logits, of size:
   *                       n_frames * batch_size * num_classes
   *
   * @return Vector of Output structs directly from the CTC decoder for additional processing.
   */
  vector<Output> decode_raw(const vector<float>& logits);

  /**
   * @brief Return character-level metadata including letter timings.
   *
   * @param logits          Flat matrix of logits, of size:
   *                        n_frames * batch_size * num_classes
   *
   * @return Metadata struct containing MetadataItem structs for each character.
   * The user is responsible for freeing Metadata by calling DS_FreeMetadata().
   */
  Metadata* decode_metadata(const vector<float>& logits);

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
  void infer(const float* mfcc, unsigned int n_frames, vector<float>& logits_output);

  void compute_mfcc(const vector<float>& audio_buffer, vector<float>& mfcc_output);
};

ModelState::ModelState()
  :
#ifndef USE_TFLITE
    mmap_env(nullptr)
  , session(nullptr)
#else // USE_TFLITE
    interpreter(nullptr)
  , fbmodel(nullptr)
#endif // USE_TFLITE
  , ncep(0)
  , ncontext(0)
  , alphabet(nullptr)
  , scorer(nullptr)
  , beam_width(0)
  , n_steps(-1)
  , n_context(-1)
  , n_features(-1)
  , mfcc_feats_per_timestep(-1)
  , sample_rate(DEFAULT_SAMPLE_RATE)
  , audio_win_len(DEFAULT_WINDOW_LENGTH)
  , audio_win_step(DEFAULT_WINDOW_STEP)
#ifdef USE_TFLITE
  , previous_state_size(0)
  , previous_state_c_(nullptr)
  , previous_state_h_(nullptr)
#endif
{
}

ModelState::~ModelState()
{
#ifndef USE_TFLITE
  if (session) {
    Status status = session->Close();
    if (!status.ok()) {
      std::cerr << "Error closing TensorFlow session: " << status << std::endl;
    }
  }
  delete mmap_env;
#endif // USE_TFLITE

  delete scorer;
  delete alphabet;
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
    while (buffer_size > 0 && audio_buffer.size() < model->audio_win_len) {
      // Convert i16 sample into f32
      float multiplier = 1.0f / (1 << 15);
      audio_buffer.push_back((float)(*buffer) * multiplier);
      ++buffer;
      --buffer_size;
    }

    // If the buffer is full, process and shift it
    if (audio_buffer.size() == model->audio_win_len) {
      processAudioWindow(audio_buffer);
      // Shift data by one step
      shift_buffer_left(audio_buffer, model->audio_win_step);
    }

    // Repeat until buffer empty
  }
}

char*
StreamingState::intermediateDecode()
{
  return model->decode(accumulated_logits);
}

char*
StreamingState::finishStream()
{
  finalizeStream();
  return model->decode(accumulated_logits);
}

Metadata*
StreamingState::finishStreamWithMetadata()
{
  finalizeStream();
  return model->decode_metadata(accumulated_logits);
}

void
StreamingState::processAudioWindow(const vector<float>& buf)
{
  // Compute MFCC features
  vector<float> mfcc;
  mfcc.reserve(model->n_features);
  model->compute_mfcc(buf, mfcc);
  pushMfccBuffer(mfcc);
}

void
StreamingState::finalizeStream()
{
  // Flush audio buffer
  processAudioWindow(audio_buffer);

  // Add empty mfcc vectors at end of sample
  for (int i = 0; i < model->n_context; ++i) {
    addZeroMfccWindow();
  }

  // Process final batch
  if (batch_buffer.size() > 0) {
    processBatch(batch_buffer, batch_buffer.size()/model->mfcc_feats_per_timestep);
  }
}

void
StreamingState::addZeroMfccWindow()
{
  vector<float> zero_buffer(model->n_features, 0.f);
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
    start = copy_up_to_n(start, end, std::back_inserter(mfcc_buffer),
                         model->mfcc_feats_per_timestep - mfcc_buffer.size());
    assert(mfcc_buffer.size() <= model->mfcc_feats_per_timestep);

    // If we have a full context window
    if (mfcc_buffer.size() == model->mfcc_feats_per_timestep) {
      processMfccWindow(mfcc_buffer);
      // Shift data by one step of one mfcc feature vector
      shift_buffer_left(mfcc_buffer, model->n_features);
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
    start = copy_up_to_n(start, end, std::back_inserter(batch_buffer),
                         model->n_steps * model->mfcc_feats_per_timestep - batch_buffer.size());
    assert(batch_buffer.size() <= model->n_steps * model->mfcc_feats_per_timestep);

    // If we have a full batch
    if (batch_buffer.size() == model->n_steps * model->mfcc_feats_per_timestep) {
      processBatch(batch_buffer, model->n_steps);
      batch_buffer.resize(0);
    }
  }
}

void
StreamingState::processBatch(const vector<float>& buf, unsigned int n_steps)
{
  model->infer(buf.data(), n_steps, accumulated_logits);
}

void
ModelState::infer(const float* aMfcc, unsigned int n_frames, vector<float>& logits_output)
{
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

#ifndef USE_TFLITE
  Tensor input(DT_FLOAT, TensorShape({BATCH_SIZE, n_steps, 2*n_context+1, n_features}));

  auto input_mapped = input.flat<float>();
  int i;
  for (i = 0; i < n_frames*mfcc_feats_per_timestep; ++i) {
    input_mapped(i) = aMfcc[i];
  }
  for (; i < n_steps*mfcc_feats_per_timestep; ++i) {
    input_mapped(i) = 0.;
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
#else // USE_TFLITE
  // Feeding input_node
  float* input_node = interpreter->typed_tensor<float>(input_node_idx);
  {
    int i;
    for (i = 0; i < n_frames*mfcc_feats_per_timestep; ++i) {
      input_node[i] = aMfcc[i];
    }
    for (; i < n_steps*mfcc_feats_per_timestep; ++i) {
      input_node[i] = 0;
    }
  }

  assert(previous_state_size > 0);

  // Feeding previous_state_c, previous_state_h
  memcpy(interpreter->typed_tensor<float>(previous_state_c_idx), previous_state_c_.get(), sizeof(float) * previous_state_size);
  memcpy(interpreter->typed_tensor<float>(previous_state_h_idx), previous_state_h_.get(), sizeof(float) * previous_state_size);

  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  float* outputs = interpreter->typed_tensor<float>(logits_idx);

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < n_frames * BATCH_SIZE * num_classes; ++t) {
    logits_output.push_back(outputs[t]);
  }

  memcpy(previous_state_c_.get(), interpreter->typed_tensor<float>(new_state_c_idx), sizeof(float) * previous_state_size);
  memcpy(previous_state_h_.get(), interpreter->typed_tensor<float>(new_state_h_idx), sizeof(float) * previous_state_size);
#endif // USE_TFLITE
}

void
ModelState::compute_mfcc(const vector<float>& samples, vector<float>& mfcc_output)
{
#ifndef USE_TFLITE
  Tensor input(DT_FLOAT, TensorShape({audio_win_len}));
  auto input_mapped = input.flat<float>();
  int i;
  for (i = 0; i < samples.size(); ++i) {
    input_mapped(i) = samples[i];
  }
  for (; i < audio_win_len; ++i) {
    input_mapped(i) = 0.f;
  }

  vector<Tensor> outputs;
  Status status = session->Run({{"input_samples", input}}, {"mfccs"}, {}, &outputs);

  if (!status.ok()) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  // The feature computation graph is hardcoded to one audio length for now
  const int n_windows = 1;
  assert(outputs[0].shape().num_elemements() / n_features == n_windows);

  auto mfcc_mapped = outputs[0].flat<float>();
  for (int i = 0; i < n_windows * n_features; ++i) {
    mfcc_output.push_back(mfcc_mapped(i));
  }
#else
  // Feeding input_node
  float* input_samples = interpreter->typed_tensor<float>(input_samples_idx);
  for (int i = 0; i < samples.size(); ++i) {
    input_samples[i] = samples[i];
  }

  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  // The feature computation graph is hardcoded to one audio length for now
  int n_windows = 1;
  TfLiteIntArray* out_dims = interpreter->tensor(mfccs_idx)->dims;
  int num_elements = 1;
  for (int i = 0; i < out_dims->size; ++i) {
    num_elements *= out_dims->data[i];
  }
  assert(num_elements / n_features == n_windows);

  float* outputs = interpreter->typed_tensor<float>(mfccs_idx);
  for (int i = 0; i < n_windows * n_features; ++i) {
    mfcc_output.push_back(outputs[i]);
  }
#endif
}

char*
ModelState::decode(const vector<float>& logits)
{
  vector<Output> out = ModelState::decode_raw(logits);
  return strdup(alphabet->LabelsToString(out[0].tokens).c_str());
}

vector<Output>
ModelState::decode_raw(const vector<float>& logits)
{
  const int cutoff_top_n = 40;
  const double cutoff_prob = 1.0;
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank
  const int n_frames = logits.size() / (BATCH_SIZE * num_classes);

  // Convert logits to double
  vector<double> inputs(logits.begin(), logits.end());

  // Vector of <probability, Output> pairs
  vector<Output> out = ctc_beam_search_decoder(
    inputs.data(), n_frames, num_classes, *alphabet, beam_width,
    cutoff_prob, cutoff_top_n, scorer);

  return out;
}

Metadata*
ModelState::decode_metadata(const vector<float>& logits)
{
  vector<Output> out = decode_raw(logits);

  std::unique_ptr<Metadata> metadata(new Metadata());
  metadata->num_items = out[0].tokens.size();
  metadata->probability = out[0].probability;

  std::unique_ptr<MetadataItem[]> items(new MetadataItem[metadata->num_items]());

  // Loop through each character
  for (int i = 0; i < out[0].tokens.size(); ++i) {
    items[i].character = strdup(alphabet->StringFromLabel(out[0].tokens[i]).c_str());
    items[i].timestep = out[0].timesteps[i];
    items[i].start_time = out[0].timesteps[i] * ((float)audio_win_step / sample_rate);

    if (items[i].start_time < 0) {
      items[i].start_time = 0;
    }
  }

  metadata->items = items.release();
  return metadata.release();
}

#ifdef USE_TFLITE
int
tflite_get_tensor_by_name(const ModelState* ctx, const vector<int>& list, const char* name)
{
  int rv = -1;

  for (int i = 0; i < list.size(); ++i) {
    const string& node_name = ctx->interpreter->tensor(list[i])->name;
    if (node_name.compare(string(name)) == 0) {
      rv = i;
    }
  }

  assert(rv >= 0);
  return rv;
}

int
tflite_get_input_tensor_by_name(const ModelState* ctx, const char* name)
{
  return ctx->interpreter->inputs()[tflite_get_tensor_by_name(ctx, ctx->interpreter->inputs(), name)];
}

int
tflite_get_output_tensor_by_name(const ModelState* ctx, const char* name)
{
  return ctx->interpreter->outputs()[tflite_get_tensor_by_name(ctx, ctx->interpreter->outputs(), name)];
}
#endif

int
DS_CreateModel(const char* aModelPath,
               unsigned int aNCep,
               unsigned int aNContext,
               const char* aAlphabetConfigPath,
               unsigned int aBeamWidth,
               ModelState** retval)
{
  std::unique_ptr<ModelState> model(new ModelState());
#ifndef USE_TFLITE
  model->mmap_env   = new MemmappedEnv(Env::Default());
#endif // USE_TFLITE
  model->ncep       = aNCep;
  model->ncontext   = aNContext;
  model->alphabet   = new Alphabet(aAlphabetConfigPath);
  model->beam_width = aBeamWidth;

  *retval = nullptr;

  DS_PrintVersions();

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, cannot continue." << std::endl;
    return DS_ERR_NO_MODEL;
  }

#ifndef USE_TFLITE
  Status status;
  SessionOptions options;

  bool is_mmap = std::string(aModelPath).find(".pbmm") != std::string::npos;
  if (!is_mmap) {
    std::cerr << "Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage." << std::endl;
  } else {
    status = model->mmap_env->InitializeFromFile(aModelPath);
    if (!status.ok()) {
      std::cerr << status << std::endl;
      return DS_ERR_FAIL_INIT_MMAP;
    }

    options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(::OptimizerOptions::L0);
    options.env = model->mmap_env;
  }

  status = NewSession(options, &model->session);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_INIT_SESS;
  }

  if (is_mmap) {
    status = ReadBinaryProto(model->mmap_env,
                             MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                             &model->graph_def);
  } else {
    status = ReadBinaryProto(Env::Default(), aModelPath, &model->graph_def);
  }
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_READ_PROTOBUF;
  }

  status = model->session->Create(model->graph_def);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_CREATE_SESS;
  }

  int graph_version = model->graph_def.version();
  if (graph_version < DS_GRAPH_VERSION) {
    std::cerr << "Specified model file version (" << graph_version << ") is "
              << "incompatible with minimum version supported by this client ("
              << DS_GRAPH_VERSION << "). See "
              << "https://github.com/mozilla/DeepSpeech/#model-compatibility "
              << "for more information" << std::endl;
    return DS_ERR_MODEL_INCOMPATIBLE;
  }

  for (int i = 0; i < model->graph_def.node_size(); ++i) {
    NodeDef node = model->graph_def.node(i);
    if (node.name() == "input_node") {
      const auto& shape = node.attr().at("shape").shape();
      model->n_steps = shape.dim(1).size();
      model->n_context = (shape.dim(2).size()-1)/2;
      model->n_features = shape.dim(3).size();
      model->mfcc_feats_per_timestep = shape.dim(2).size() * shape.dim(3).size();
    } else if (node.name() == "logits_shape") {
      Tensor logits_shape = Tensor(DT_INT32, TensorShape({3}));
      if (!logits_shape.FromProto(node.attr().at("value").tensor())) {
        continue;
      }

      int final_dim_size = logits_shape.vec<int>()(2) - 1;
      if (final_dim_size != model->alphabet->GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << model->alphabet->GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        return DS_ERR_INVALID_ALPHABET;
      }
    } else if (node.name() == "model_metadata") {
      int sample_rate = node.attr().at("sample_rate").i();
      model->sample_rate = sample_rate;
      int win_len_ms = node.attr().at("feature_win_len").i();
      int win_step_ms = node.attr().at("feature_win_step").i();
      model->audio_win_len = sample_rate * (win_len_ms / 1000.0);
      model->audio_win_step = sample_rate * (win_step_ms / 1000.0);
    }
  }

  if (model->n_context == -1 || model->n_features == -1) {
    std::cerr << "Error: Could not infer input shape from model file. "
              << "Make sure input_node is a 4D tensor with shape "
              << "[batch_size=1, time, window_size, n_features]."
              << std::endl;
    return DS_ERR_INVALID_SHAPE;
  }

  *retval = model.release();
  return DS_ERR_OK;
#else // USE_TFLITE
  model->fbmodel = tflite::FlatBufferModel::BuildFromFile(aModelPath);
  if (!model->fbmodel) {
    std::cerr << "Error at reading model file " << aModelPath << std::endl;
    return DS_ERR_FAIL_INIT_MMAP;
  }


  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model->fbmodel, resolver)(&model->interpreter);
  if (!model->interpreter) {
    std::cerr << "Error at InterpreterBuilder for model file " << aModelPath << std::endl;
    return DS_ERR_FAIL_INTERPRETER;
  }

  model->interpreter->AllocateTensors();
  model->interpreter->SetNumThreads(4);

  // Query all the index once
  model->input_node_idx       = tflite_get_input_tensor_by_name(model.get(), "input_node");
  model->previous_state_c_idx = tflite_get_input_tensor_by_name(model.get(), "previous_state_c");
  model->previous_state_h_idx = tflite_get_input_tensor_by_name(model.get(), "previous_state_h");
  model->input_samples_idx    = tflite_get_input_tensor_by_name(model.get(), "input_samples");
  model->logits_idx           = tflite_get_output_tensor_by_name(model.get(), "logits");
  model->new_state_c_idx      = tflite_get_output_tensor_by_name(model.get(), "new_state_c");
  model->new_state_h_idx      = tflite_get_output_tensor_by_name(model.get(), "new_state_h");
  model->mfccs_idx            = tflite_get_output_tensor_by_name(model.get(), "mfccs");

  TfLiteIntArray* dims_input_node = model->interpreter->tensor(model->input_node_idx)->dims;

  model->n_steps = dims_input_node->data[1];
  model->n_context = (dims_input_node->data[2] - 1 ) / 2;
  model->n_features = dims_input_node->data[3];
  model->mfcc_feats_per_timestep = dims_input_node->data[2] * dims_input_node->data[3];

  TfLiteIntArray* dims_logits = model->interpreter->tensor(model->logits_idx)->dims;
  const int final_dim_size = dims_logits->data[1] - 1;
  if (final_dim_size != model->alphabet->GetSize()) {
    std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
              << "has size " << model->alphabet->GetSize()
              << ", but model has " << final_dim_size
              << " classes in its output. Make sure you're passing an alphabet "
              << "file with the same size as the one used for training."
              << std::endl;
    return DS_ERR_INVALID_ALPHABET;
  }

  TfLiteIntArray* dims_c = model->interpreter->tensor(model->previous_state_c_idx)->dims;
  TfLiteIntArray* dims_h = model->interpreter->tensor(model->previous_state_h_idx)->dims;
  assert(dims_c->data[1] == dims_h->data[1]);

  model->previous_state_size = dims_c->data[1];
  model->previous_state_c_.reset(new float[model->previous_state_size]());
  model->previous_state_h_.reset(new float[model->previous_state_size]());

  // Set initial values for previous_state_c and previous_state_h
  memset(model->previous_state_c_.get(), 0, sizeof(float) * model->previous_state_size);
  memset(model->previous_state_h_.get(), 0, sizeof(float) * model->previous_state_size);

  *retval = model.release();
  return DS_ERR_OK;
#endif // USE_TFLITE
}

void
DS_DestroyModel(ModelState* ctx)
{
  delete ctx;
}

int
DS_EnableDecoderWithLM(ModelState* aCtx,
                       const char* aAlphabetConfigPath,
                       const char* aLMPath,
                       const char* aTriePath,
                       float aLMAlpha,
                       float aLMBeta)
{
  try {
    aCtx->scorer = new Scorer(aLMAlpha, aLMBeta,
                              aLMPath ? aLMPath : "",
                              aTriePath ? aTriePath : "",
                              *aCtx->alphabet);
    return DS_ERR_OK;
  } catch (...) {
    return DS_ERR_INVALID_LM;
  }
}

int
DS_SetupStream(ModelState* aCtx,
               unsigned int aPreAllocFrames,
               unsigned int aSampleRate,
               StreamingState** retval)
{
  *retval = nullptr;

#ifndef USE_TFLITE
  Status status = aCtx->session->Run({}, {}, {"initialize_state"}, nullptr);
  if (!status.ok()) {
    std::cerr << "Error running session: " << status << std::endl;
    return DS_ERR_FAIL_RUN_SESS;
  }
#endif // USE_TFLITE

  std::unique_ptr<StreamingState> ctx(new StreamingState());
  if (!ctx) {
    std::cerr << "Could not allocate streaming state." << std::endl;
    return DS_ERR_FAIL_CREATE_STREAM;
  }

  const size_t num_classes = aCtx->alphabet->GetSize() + 1; // +1 for blank

  // Default initial allocation = 3 seconds.
  if (aPreAllocFrames == 0) {
    aPreAllocFrames = 150;
  }

  ctx->accumulated_logits.reserve(aPreAllocFrames * BATCH_SIZE * num_classes);

  ctx->audio_buffer.reserve(aCtx->audio_win_len);
  ctx->mfcc_buffer.reserve(aCtx->mfcc_feats_per_timestep);
  ctx->mfcc_buffer.resize(aCtx->n_features*aCtx->n_context, 0.f);
  ctx->batch_buffer.reserve(aCtx->n_steps * aCtx->mfcc_feats_per_timestep);

  ctx->model = aCtx;

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
DS_IntermediateDecode(StreamingState* aSctx)
{
  return aSctx->intermediateDecode();
}

char*
DS_FinishStream(StreamingState* aSctx)
{
  char* str = aSctx->finishStream();
  DS_DiscardStream(aSctx);
  return str;
}

Metadata*
DS_FinishStreamWithMetadata(StreamingState* aSctx)
{
  Metadata* metadata = aSctx->finishStreamWithMetadata();
  DS_DiscardStream(aSctx);
  return metadata;
}

StreamingState*
SetupStreamAndFeedAudioContent(ModelState* aCtx,
                               const short* aBuffer,
                               unsigned int aBufferSize,
                               unsigned int aSampleRate)
{
  StreamingState* ctx;
  int status = DS_SetupStream(aCtx, 0, aSampleRate, &ctx);
  if (status != DS_ERR_OK) {
    return nullptr;
  }
  DS_FeedAudioContent(ctx, aBuffer, aBufferSize);
  return ctx;
}

char*
DS_SpeechToText(ModelState* aCtx,
                const short* aBuffer,
                unsigned int aBufferSize,
                unsigned int aSampleRate)
{
  StreamingState* ctx = SetupStreamAndFeedAudioContent(aCtx, aBuffer, aBufferSize, aSampleRate);
  return DS_FinishStream(ctx);
}

Metadata*
DS_SpeechToTextWithMetadata(ModelState* aCtx,
                            const short* aBuffer,
                            unsigned int aBufferSize,
                            unsigned int aSampleRate)
{
  StreamingState* ctx = SetupStreamAndFeedAudioContent(aCtx, aBuffer, aBufferSize, aSampleRate);
  return DS_FinishStreamWithMetadata(ctx);
}

void
DS_DiscardStream(StreamingState* aSctx)
{
  delete aSctx;
}

void
DS_FreeMetadata(Metadata* m)
{
  if (m) {
    for (int i = 0; i < m->num_items; ++i) {
      free(m->items[i].character);
    }
    delete[] m->items;
    delete m;
  }
}

void
DS_PrintVersions() {
  std::cerr << "TensorFlow: " << tf_local_git_version() << std::endl;
  std::cerr << "DeepSpeech: " << ds_git_version() << std::endl;
#ifdef __ANDROID__
  LOGE("TensorFlow: %s", tf_local_git_version());
  LOGD("TensorFlow: %s", tf_local_git_version());
  LOGE("DeepSpeech: %s", ds_git_version());
  LOGD("DeepSpeech: %s", ds_git_version());
#endif
}

