#ifdef DS_NATIVE_MODEL
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "native_client/deepspeech_model_core.h" // generated
#endif

#include <iostream>

#include "deepspeech.h"
#include "deepspeech_utils.h"
#include "alphabet.h"
#include "beam_search.h"

#include "tensorflow/core/public/version.h"
#include "native_client/ds_version.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#define BATCH_SIZE 1

using namespace tensorflow;
using tensorflow::ctc::CTCBeamSearchDecoder;
using tensorflow::ctc::CTCDecoder;

namespace DeepSpeech {

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
};

DEEPSPEECH_EXPORT
Model::Model(const char* aModelPath, int aNCep, int aNContext,
             const char* aAlphabetConfigPath, int aBeamWidth)
{
  mPriv             = new Private;
  mPriv->mmap_env   = new MemmappedEnv(Env::Default());
  mPriv->session    = NULL;
  mPriv->scorer     = NULL;
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
  }

  if (is_mmap) {
    status = mPriv->mmap_env->InitializeFromFile(aModelPath);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      return;
    }

    options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(::OptimizerOptions::L0);
    options.env = mPriv->mmap_env;
  }

  status = NewSession(options, &mPriv->session);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
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
    mPriv->session->Close();
    mPriv->session = NULL;
    std::cerr << status.ToString() << std::endl;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = NULL;
    std::cerr << status.ToString() << std::endl;
    return;
  }

  for (int i = 0; i < mPriv->graph_def.node_size(); ++i) {
    NodeDef node = mPriv->graph_def.node(i);
    if (node.name() == "logits/shape/2") {
      int final_dim_size = node.attr().at("value").tensor().int_val(0) - 1;
      if (final_dim_size != mPriv->alphabet->GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << mPriv->alphabet->GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        mPriv->session->Close();
        mPriv->session = NULL;
        return;
      }
      break;
    }
  }
}

DEEPSPEECH_EXPORT
Model::~Model()
{
  if (mPriv->session) {
    mPriv->session->Close();
  }

  delete mPriv->mmap_env;
  delete mPriv->alphabet;
  delete mPriv->scorer;

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

char*
Model::decode(int aNFrames, float*** aLogits)
{
  const int batch_size = BATCH_SIZE;
  const int top_paths = 1;
  const int timesteps = aNFrames;
  const size_t num_classes = mPriv->alphabet->GetSize() + 1; // +1 for blank

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&aLogits[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  // CTCDecoder::Output is std::vector<std::vector<int>>
  std::vector<CTCDecoder::Output> decoder_outputs(top_paths);
  for (CTCDecoder::Output& output : decoder_outputs) {
    output.resize(batch_size);
  }
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  if (mPriv->scorer == NULL) {
    CTCBeamSearchDecoder<>::DefaultBeamScorer scorer;
    CTCBeamSearchDecoder<> decoder(num_classes,
                                   mPriv->beam_width,
                                   &scorer,
                                   batch_size);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  } else {
    CTCBeamSearchDecoder<KenLMBeamState> decoder(num_classes,
                                                 mPriv->beam_width,
                                                 mPriv->scorer,
                                                 batch_size);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  }

  // Output is an array of shape (1, n_results, result_length).
  // In this case, n_results is also equal to 1.
  size_t output_length = decoder_outputs[0][0].size() + 1;

  size_t decoded_length = 1; // add 1 for the \0
  for (int i = 0; i < output_length - 1; i++) {
    int64 character = decoder_outputs[0][0][i];
    const std::string& str = mPriv->alphabet->StringFromLabel(character);
    decoded_length += str.size();
  }

  char* output = (char*)malloc(sizeof(char) * decoded_length);
  char* pen = output;
  for (int i = 0; i < output_length - 1; i++) {
    int64 character = decoder_outputs[0][0][i];
    const std::string& str = mPriv->alphabet->StringFromLabel(character);
    strncpy(pen, str.c_str(), str.size());
    pen += str.size();
  }
  *pen = '\0';

  for (int i = 0; i < timesteps; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      free(aLogits[i][j]);
    }
    free(aLogits[i]);
  }
  free(aLogits);

  return output;
}

DEEPSPEECH_EXPORT
char*
Model::infer(float* aMfcc, int aNFrames, int aFrameLen)
{
  const int batch_size = BATCH_SIZE;
  const int timesteps = aNFrames;
  const size_t num_classes = mPriv->alphabet->GetSize() + 1; // +1 for blank

  const int frameSize = mPriv->ncep + (2 * mPriv->ncep * mPriv->ncontext);

  float*** input_data_mat = (float***)calloc(timesteps, sizeof(float**));
  for (int i = 0; i < timesteps; ++i) {
    input_data_mat[i] = (float**)calloc(batch_size, sizeof(float*));
    for (int j = 0; j < batch_size; ++j) {
      input_data_mat[i][j] = (float*)calloc(num_classes, sizeof(float));
    }
  }

  if (mPriv->run_aot) {
#ifdef DS_NATIVE_MODEL
    Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

    nativeModel nm(nativeModel::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);
    nm.set_thread_pool(&device);

    for (int ot = 0; ot < timesteps; ot += DS_MODEL_TIMESTEPS) {
      nm.set_arg0_data(&(aMfcc[ot * frameSize]));
      nm.Run();

      // The CTCDecoder works with log-probs.
      for (int t = 0; t < DS_MODEL_TIMESTEPS, (ot + t) < timesteps; ++t) {
        for (int b = 0; b < batch_size; ++b) {
          for (int c = 0; c < num_classes; ++c) {
            input_data_mat[ot + t][b][c] = nm.result0(t, b, c);
          }
        }
      }
    }
#else
    std::cerr << "No support for native model built-in." << std::endl;
    return NULL;
#endif // DS_NATIVE_MODEL
  } else {
    if (aFrameLen == 0) {
      aFrameLen = frameSize;
    } else if (aFrameLen < frameSize) {
      std::cerr << "mfcc features array is too small (expected " <<
        frameSize << ", got " << aFrameLen << ")\n";
      return NULL;
    }

    Tensor input(DT_FLOAT, TensorShape({1, aNFrames, frameSize}));

    auto input_mapped = input.tensor<float, 3>();
    for (int i = 0, idx = 0; i < aNFrames; i++) {
      for (int j = 0; j < frameSize; j++, idx++) {
        input_mapped(0, i, j) = aMfcc[idx];
      }
      idx += (aFrameLen - frameSize);
    }

    Tensor n_frames(DT_INT32, TensorShape({1}));
    n_frames.scalar<int>()() = aNFrames;

    // The CTC Beam Search decoder takes logits as input, we can feed those from
    // the "logits" node in official models or
    // the "logits_output_node" in old AOT hacking models
    std::vector<Tensor> outputs;
    Status status = mPriv->session->Run(
      {{ "input_node", input }, { "input_lengths", n_frames }},
      {"logits"}, {}, &outputs);

    // If "logits" doesn't exist, this is an older graph. Try to recover.
    if (status.code() == tensorflow::error::NOT_FOUND) {
      status.IgnoreError();
      status = mPriv->session->Run(
        {{ "input_node", input }, { "input_lengths", n_frames }},
        {"logits_output_node"}, {}, &outputs);
    }

    if (!status.ok()) {
      std::cerr << "Error running session: " << status.ToString() << "\n";
      return NULL;
    }

    auto logits_mapped = outputs[0].tensor<float, 3>();
    // The CTCDecoder works with log-probs.
    for (int t = 0; t < timesteps; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
          input_data_mat[t][b][c] = logits_mapped(t, b, c);
        }
      }
    }
  }

  return decode(aNFrames, input_data_mat);
}

DEEPSPEECH_EXPORT
char*
Model::stt(const short* aBuffer, unsigned int aBufferSize, int aSampleRate)
{
  float* mfcc;
  char* string;
  int n_frames;

  getInputVector(aBuffer, aBufferSize, aSampleRate, &mfcc, &n_frames, NULL);
  string = infer(mfcc, n_frames);
  free(mfcc);
  return string;
}

DEEPSPEECH_EXPORT
void
print_versions() {
  std::cerr << "TensorFlow: " << tf_git_version() << std::endl;
  std::cerr << "DeepSpeech: " << ds_git_version() << std::endl;
}

}
