#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "deepspeech.h"
#include "c_speech_features.h"

#define COEFF 0.97f
#define WIN_LEN 0.025f
#define WIN_STEP 0.01f
#define N_FFT 512
#define N_FILTERS 26
#define LOWFREQ 0
#define N_CEP 26
#define CEP_LIFTER 22
#define N_CONTEXT 9

using namespace tensorflow;

struct _DeepSpeechContext {
  Session* session;
  GraphDef graph_def;
  int ncep;
  int ncontext;
};

DeepSpeechContext*
DsInit(const char* aModelPath, int aNCep, int aNContext)
{
  if (!aModelPath) {
    return NULL;
  }

  DeepSpeechContext* ctx = new DeepSpeechContext;

  Status status = NewSession(SessionOptions(), &ctx->session);
  if (!status.ok()) {
    delete ctx;
    return NULL;
  }

  status = ReadBinaryProto(Env::Default(), aModelPath, &ctx->graph_def);
  if (!status.ok()) {
    ctx->session->Close();
    delete ctx;
    return NULL;
  }

  status = ctx->session->Create(ctx->graph_def);
  if (!status.ok()) {
    ctx->session->Close();
    delete ctx;
    return NULL;
  }

  ctx->ncep = aNCep;
  ctx->ncontext = aNContext;

  return ctx;
}

void
DsClose(DeepSpeechContext* aCtx)
{
  if (!aCtx) {
    return;
  }

  aCtx->session->Close();
  delete aCtx;
}

int
DsGetMfccFrames(DeepSpeechContext* aCtx, const short* aBuffer,
                size_t aBufferSize, int aSampleRate, float** aMfcc)
{
  const int contextSize = aCtx->ncep * aCtx->ncontext;
  const int frameSize = aCtx->ncep + (2 * aCtx->ncep * aCtx->ncontext);

  // Compute MFCC features
  float* mfcc;
  int n_frames = csf_mfcc(aBuffer, aBufferSize, aSampleRate,
                          WIN_LEN, WIN_STEP, aCtx->ncep, N_FILTERS, N_FFT,
                          LOWFREQ, aSampleRate/2, COEFF, CEP_LIFTER, 1, NULL,
                          &mfcc);

  // Take every other frame (BiRNN stride of 2) and add past/future context
  int ds_input_length = (n_frames + 1) / 2;
  // TODO: Use MFCC of silence instead of zero
  float* ds_input = (float*)calloc(sizeof(float), ds_input_length * frameSize);
  for (int i = 0, idx = 0, mfcc_idx = 0; i < ds_input_length;
       i++, idx += frameSize, mfcc_idx += aCtx->ncep * 2) {
    // Past context
    for (int j = N_CONTEXT; j > 0; j--) {
      int frame_index = (i * 2) - (j * 2);
      if (frame_index < 0) { continue; }
      int mfcc_base = frame_index * aCtx->ncep;
      int base = (N_CONTEXT - j) * N_CEP;
      for (int k = 0; k < N_CEP; k++) {
        ds_input[idx + base + k] = mfcc[mfcc_base + k];
      }
    }

    // Present context
    for (int j = 0; j < N_CEP; j++) {
      ds_input[idx + j + contextSize] = mfcc[mfcc_idx + j];
    }

    // Future context
    for (int j = 1; j <= N_CONTEXT; j++) {
      int frame_index = (i * 2) + (j * 2);
      if (frame_index >= n_frames) { continue; }
      int mfcc_base = frame_index * aCtx->ncep;
      int base = contextSize + N_CEP + ((j - 1) * N_CEP);
      for (int k = 0; k < N_CEP; k++) {
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

  *aMfcc = ds_input;
  return ds_input_length;
}

char*
DsInfer(DeepSpeechContext* aCtx, float* aMfcc, int aNFrames)
{
  const int frameSize = aCtx->ncep + (2 * aCtx->ncep * aCtx->ncontext);
  Tensor input(DT_FLOAT, TensorShape({1, aNFrames, frameSize}));

  auto input_mapped = input.tensor<float, 3>();
  for (int i = 0, idx = 0; i < aNFrames; i++) {
    for (int j = 0; j < frameSize; j++, idx++) {
      input_mapped(0, i, j) = aMfcc[idx];
    }
  }

  Tensor n_frames(DT_INT32, TensorShape({1}));
  n_frames.scalar<int>()() = aNFrames;

  std::vector<Tensor> outputs;
  Status status =
    aCtx->session->Run({{ "input_node", input }, { "input_lengths", n_frames }},
                       {"output_node"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Error running session: " << status.ToString() << "\n";
    return NULL;
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
DsSTT(DeepSpeechContext* aCtx, const short* aBuffer, size_t aBufferSize,
      int aSampleRate)
{
  float* mfcc;
  char* string;
  int n_frames =
    DsGetMfccFrames(aCtx, aBuffer, aBufferSize, aSampleRate, &mfcc);
  string = DsInfer(aCtx, mfcc, n_frames);
  free(mfcc);
  return string;
}
