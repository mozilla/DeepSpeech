#include "deepspeech_utils.h"
#include "c_speech_features.h"
#include <stdlib.h>

#define COEFF 0.97f
#define WIN_LEN 0.025f
#define WIN_STEP 0.01f
#define N_FFT 512
#define N_FILTERS 26
#define LOWFREQ 0
#define CEP_LIFTER 22

namespace DeepSpeech {

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
                          WIN_LEN, WIN_STEP, aNCep, N_FILTERS, N_FFT,
                          LOWFREQ, aSampleRate/2, COEFF, CEP_LIFTER, 1, NULL,
                          &mfcc);

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
    *aFrameLen = frameSize;
  }
}

}
