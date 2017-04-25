#include "deepspeech.h"
#include "c_speech_features.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#define COEFF 0.97f
#define WIN_LEN 0.025f
#define WIN_STEP 0.01f
#define N_FFT 512
#define N_FILTERS 26
#define LOWFREQ 0
#define CEP_LIFTER 22

using namespace tensorflow;

struct _DeepSpeechPrivate {
  Session* session;
  GraphDef graph_def;
  int ncep;
  int ncontext;
};

DeepSpeech::DeepSpeech(const char* aModelPath, int aNCep, int aNContext)
{
  mPriv = new DeepSpeechPrivate;

  if (!aModelPath) {
    return;
  }

  Status status = NewSession(SessionOptions(), &mPriv->session);
  if (!status.ok()) {
    return;
  }

  status = ReadBinaryProto(Env::Default(), aModelPath, &mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = nullptr;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = nullptr;
    return;
  }

  mPriv->ncep = aNCep;
  mPriv->ncontext = aNContext;
}

DeepSpeech::~DeepSpeech()
{
  if (mPriv->session) {
    mPriv->session->Close();
  }

  delete mPriv;
}

void
DeepSpeech::getMfccFrames(const short* aBuffer, unsigned int aBufferSize,
                          int aSampleRate, float** aMfcc, int* aNFrames,
                          int* aFrameLen)
{
  const int contextSize = mPriv->ncep * mPriv->ncontext;
  const int frameSize = mPriv->ncep + (2 * mPriv->ncep * mPriv->ncontext);

  // Compute MFCC features
  float* mfcc;
  int n_frames = csf_mfcc(aBuffer, aBufferSize, aSampleRate,
                          WIN_LEN, WIN_STEP, mPriv->ncep, N_FILTERS, N_FFT,
                          LOWFREQ, aSampleRate/2, COEFF, CEP_LIFTER, 1, NULL,
                          &mfcc);

  // Take every other frame (BiRNN stride of 2) and add past/future context
  int ds_input_length = (n_frames + 1) / 2;
  // TODO: Use MFCC of silence instead of zero
  float* ds_input = (float*)calloc(sizeof(float), ds_input_length * frameSize);
  for (int i = 0, idx = 0, mfcc_idx = 0; i < ds_input_length;
       i++, idx += frameSize, mfcc_idx += mPriv->ncep * 2) {
    // Past context
    for (int j = mPriv->ncontext; j > 0; j--) {
      int frame_index = (i * 2) - (j * 2);
      if (frame_index < 0) { continue; }
      int mfcc_base = frame_index * mPriv->ncep;
      int base = (mPriv->ncontext - j) * mPriv->ncep;
      for (int k = 0; k < mPriv->ncep; k++) {
        ds_input[idx + base + k] = mfcc[mfcc_base + k];
      }
    }

    // Present context
    for (int j = 0; j < mPriv->ncep; j++) {
      ds_input[idx + j + contextSize] = mfcc[mfcc_idx + j];
    }

    // Future context
    for (int j = 1; j <= mPriv->ncontext; j++) {
      int frame_index = (i * 2) + (j * 2);
      if (frame_index >= n_frames) { continue; }
      int mfcc_base = frame_index * mPriv->ncep;
      int base = contextSize + mPriv->ncep + ((j - 1) * mPriv->ncep);
      for (int k = 0; k < mPriv->ncep; k++) {
        ds_input[idx + base + k] = mfcc[mfcc_base + k];
      }
    }
  }

  // Free mfcc array
  free(mfcc);

  // Whiten inputs (TODO: Should we whiten)
  double n_inputs = (double)(ds_input_length * frameSize);
  double mean = 0.0;
  for (int idx = 0; idx < n_inputs; idx++) {
    mean += ds_input[idx] / n_inputs;
  }

  double stddev = 0.0;
  for (int idx = 0; idx < n_inputs; idx++) {
    stddev += pow(fabs(ds_input[idx] - mean), 2.0) / n_inputs;
  }
  stddev = sqrt(stddev);

  for (int idx = 0; idx < n_inputs; idx++) {
    ds_input[idx] = (float)((ds_input[idx] - mean) / stddev);
  }

  if (aMfcc) {
    *aMfcc = ds_input;
  }
  if (aNFrames) {
    *aNFrames = ds_input_length;
  }
  if (aFrameLen) {
    *aFrameLen = contextSize;
  }
}

char*
DeepSpeech::infer(float* aMfcc, int aNFrames, int aFrameLen)
{
  if (!mPriv->session) {
    return nullptr;
  }

  const int frameSize = mPriv->ncep + (2 * mPriv->ncep * mPriv->ncontext);
  if (aFrameLen == 0) {
    aFrameLen = frameSize;
  } else if (aFrameLen < frameSize) {
    std::cerr << "mfcc features array is too small (expected " <<
      frameSize << ", got " << aFrameLen << ")\n";
    return nullptr;
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

  std::vector<Tensor> outputs;
  Status status = mPriv->session->Run(
    {{ "input_node", input }, { "input_lengths", n_frames }},
    {"output_node"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Error running session: " << status.ToString() << "\n";
    return nullptr;
  }

  // Output is an array of shape (1, n_results, result_length).
  // In this case, n_results is also equal to 1.
  auto output_mapped = outputs[0].tensor<int64, 3>();
  int length = output_mapped.dimension(2) + 1;
  char* output = (char*)malloc(sizeof(char) * length);
  for (int i = 0; i < length - 1; i++) {
    int64 character = output_mapped(0, 0, i);
    output[i] = (character ==  0) ? ' ' : (character + 'a' - 1);
  }
  output[length - 1] = '\0';

  return output;
}

char*
DeepSpeech::stt(const short* aBuffer, unsigned int aBufferSize, int aSampleRate)
{
  float* mfcc;
  char* string;
  int n_frames;

  getMfccFrames(aBuffer, aBufferSize, aSampleRate, &mfcc, &n_frames, nullptr);
  string = infer(mfcc, n_frames);
  free(mfcc);
  return string;
}
