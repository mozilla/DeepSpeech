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
                size_t aBufferSize, int aSampleRate, float*** aMfcc)
{
  const int contextSize = aCtx->ncep * aCtx->ncontext;
  const int frameSize = aCtx->ncep + (2 * aCtx->ncep * aCtx->ncontext);

  // Compute MFCC features
  float** mfcc;
  int n_frames = csf_mfcc(aBuffer, aBufferSize, aSampleRate,
                          WIN_LEN, WIN_STEP, aCtx->ncep, N_FILTERS, N_FFT,
                          LOWFREQ, aSampleRate/2, COEFF, CEP_LIFTER, 1, &mfcc);

  // Take every other frame (BiRNN stride of 2) and add past/future context
  int ds_input_length = (n_frames + 1) / 2;
  float** ds_input = (float**)malloc(sizeof(float*) * ds_input_length);
  for (int i = 0; i < ds_input_length; i++) {
    // TODO: Use MFCC of silence instead of zero
    ds_input[i] = (float*)calloc(sizeof(float), frameSize);

    // Past context
    for (int j = N_CONTEXT; j > 0; j--) {
      int frame_index = (i * 2) - (j * 2);
      if (frame_index < 0) { continue; }
      int base = (N_CONTEXT - j) * N_CEP;
      for (int k = 0; k < N_CEP; k++) {
        ds_input[i][base + k] = mfcc[frame_index][k];
      }
    }

    // Present context
    for (int j = 0; j < N_CEP; j++) {
      ds_input[i][j + contextSize] = mfcc[i * 2][j];
    }

    // Future context
    for (int j = 1; j <= N_CONTEXT; j++) {
      int frame_index = (i * 2) + (j * 2);
      if (frame_index >= n_frames) { continue; }
      int base = contextSize + N_CEP + ((j - 1) * N_CEP);
      for (int k = 0; k < N_CEP; k++) {
        ds_input[i][base + k] = mfcc[frame_index][k];
      }
    }
  }

  // Free mfcc array
  for (int i = 0; i < n_frames; i++) {
    free(mfcc[i]);
  }
  free(mfcc);

  // Whiten inputs (TODO: Should we whiten)
  double n_inputs = (double)(ds_input_length * frameSize);
  double mean = 0.0;
  for (int i = 0; i < ds_input_length; i++) {
    for (int j = 0; j < frameSize; j++) {
      mean += ds_input[i][j] / n_inputs;
    }
  }

  double stddev = 0.0;
  for (int i = 0; i < ds_input_length; i++) {
    for (int j = 0; j < frameSize; j++) {
      stddev += pow(fabs(ds_input[i][j] - mean), 2.0) / n_inputs;
    }
  }
  stddev = sqrt(stddev);

  for (int i = 0; i < ds_input_length; i++) {
    for (int j = 0; j < frameSize; j++) {
      ds_input[i][j] = (float)((ds_input[i][j] - mean) / stddev);
    }
  }

  *aMfcc = ds_input;
  return ds_input_length;
}

void
DsFreeMfccFrames(float** aMfcc, int aNFrames)
{
  for (int i = 0; i < aNFrames; i++) {
    free(aMfcc[i]);
  }
  free(aMfcc);
}

char*
DsInfer(DeepSpeechContext* aCtx, float** aMfcc, int aNFrames)
{
  const int frameSize = aCtx->ncep + (2 * aCtx->ncep * aCtx->ncontext);
  Tensor input(DT_FLOAT, TensorShape({1, aNFrames, frameSize}));

  auto input_mapped = input.tensor<float, 3>();
  for (int i = 0; i < aNFrames; i++) {
    for (int j = 0; j < frameSize; j++) {
      input_mapped(0, i, j) = aMfcc[i][j];
    }
  }

  std::vector<Tensor> outputs;
  Status status =
    aCtx->session->Run({{ "input_node", input }},
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
  float** mfcc;
  char* string;
  int n_frames =
    DsGetMfccFrames(aCtx, aBuffer, aBufferSize, aSampleRate, &mfcc);
  string = DsInfer(aCtx, mfcc, n_frames);
  DsFreeMfccFrames(mfcc, n_frames);
  return string;
}
