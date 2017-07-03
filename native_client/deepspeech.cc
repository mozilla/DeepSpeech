#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "deepspeech.h"
#include "deepspeech_utils.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/ctc/ctc_beam_search.h"

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if defined(TF_HAS_NATIVE_MODEL)
#include "native_client/deepspeech_model.h" // generated
#endif

using namespace tensorflow;

using tensorflow::ctc::CTCBeamSearchDecoder;
using tensorflow::ctc::CTCDecoder;

namespace DeepSpeech {

class Private {
  public:
    Session* session;
    GraphDef graph_def;
    int ncep;
    int ncontext;
};

Model::Model(const char* aModelPath, int aNCep, int aNContext)
{
  mPriv = new Private;
  mPriv->ncep     = aNCep;
  mPriv->ncontext = aNContext;
  mPriv->session  = NULL;

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, will rely on built-in model." << std::endl;
    return;
  }

  Status status = NewSession(SessionOptions(), &mPriv->session);
  if (!status.ok()) {
    return;
  }

  status = ReadBinaryProto(Env::Default(), aModelPath, &mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = NULL;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = NULL;
    return;
  }
}

Model::~Model()
{
  if (mPriv->session) {
    mPriv->session->Close();
  }

  delete mPriv;
}

void
Model::getInputVector(const short* aBuffer, unsigned int aBufferSize,
                           int aSampleRate, float** aMfcc, int* aNFrames,
                           int* aFrameLen)
{
  return audioToInputVector(aBuffer, aBufferSize, aSampleRate, mPriv->ncep,
                            mPriv->ncontext, aMfcc, aNFrames, aFrameLen);
}

char*
Model::decode(int aNFrames, float input_data_mat[][1][N_CHARACTERS])
{
/*
const int64 max_time = inputs_shape.dim_size(0);
const int64 batch_size = inputs_shape.dim_size(1);
const int64 num_classes_raw = inputs_shape.dim_size(2);
*/

  const int batch_size = 1;
  const int top_paths = 1;
  const int timesteps = aNFrames;
  const int num_classes = N_CHARACTERS;

  CTCBeamSearchDecoder<>::DefaultBeamScorer default_scorer;
  CTCBeamSearchDecoder<> decoder(num_classes, 100 /* beam_width=100 */, &default_scorer);

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  // CTCDecoder::Output is std::vector<std::vector<int>>
  std::vector<CTCDecoder::Output> outputs(top_paths);
  for (CTCDecoder::Output& output : outputs) {
    output.resize(batch_size);
  }
  float score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

  decoder.Decode(seq_len, inputs, &outputs, &scores).ok();

  int length = outputs[0][0].size() + 1;
  char* output = (char*)malloc(sizeof(char) * length);
  for (int i = 0; i < length - 1; i++) {
    int64 character = outputs[0][0][i];
    output[i] = (character ==  0) ? ' ' : (character + 'a' - 1);
  }
  output[length - 1] = '\0';

  return output;
}

char*
Model::infer(float* aMfcc, int aNFrames, int aFrameLen)
{
  const int batch_size = 1;
  const int timesteps = aNFrames;
  const int num_classes = N_CHARACTERS;

  float input_data_mat[timesteps][batch_size][num_classes];

  if (!mPriv->session) {
#if defined(TF_HAS_NATIVE_MODEL)
    Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

    nativeModel nm;
    nm.set_thread_pool(&device);

    nm.set_arg0_data(aMfcc);
    nm.Run();

    // The CTCDecoder works with log-probs.
    for (int t = 0; t < timesteps; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
          input_data_mat[t][b][c] = nm.result0(t, b, c);
        }
      }
    }
#else
    std::cerr << "No support for native model built-in.";
    return NULL;
#endif // TF_HAS_NATIVE_MODEL
  } else {
    const int frameSize = mPriv->ncep + (2 * mPriv->ncep * mPriv->ncontext);
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

    std::vector<Tensor> logits;
    Status status = mPriv->session->Run(
      {{ "input_node", input }, { "input_lengths", n_frames }},
      {"logits_output_node"}, {}, &logits);
    if (!status.ok()) {
      std::cerr << "Error running session: " << status.ToString() << "\n";
      return NULL;
    }

    auto logits_mapped = logits[0].tensor<float, 3>();
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

}
