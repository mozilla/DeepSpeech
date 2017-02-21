#include <float.h>
#include "c_speech_features.h"
#include "tools/kiss_fftr.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

int
csf_mfcc(const short* aSignal, unsigned int aSignalLen, int aSampleRate,
         float aWinLen, float aWinStep, int aNCep, int aNFilters, int aNFFT,
         int aLowFreq, int aHighFreq, float aPreemph, int aCepLifter,
         int aAppendEnergy, float*** aMFCC)
{
  int i, j, k;
  float** feat;
  float* energy;

  int n_frames = csf_fbank(aSignal, aSignalLen, aSampleRate, aWinLen, aWinStep,
                           aNFilters, aNFFT, aLowFreq, aHighFreq, aPreemph,
                           &feat, &energy);

  // Perform DCT-II
  float sf1 = sqrtf(1 / (4 * (float)aNFilters));
  float sf2 = sqrtf(1 / (2 * (float)aNFilters));
  float** mfcc = (float**)malloc(sizeof(float*) * n_frames);
  for (i = 0; i < n_frames; i++) {
    for (j = 0; j < aNFilters; j++) {
      feat[i][j] = logf(feat[i][j]);
    }
    mfcc[i] = (float*)calloc(sizeof(float), aNCep);
    for (j = 0; j < aNCep; j++) {
      for (k = 0; k < aNCep; k++) {
        mfcc[i][j] += feat[i][k] *
          cosf(M_PI * j * (2 * k + 1) / (2 * aNFilters));
      }
      mfcc[i][j] *= 2 * ((i == 0 && j == 0) ? sf1 : sf2);
    }
  }

  // Apply a cepstral lifter
  csf_lifter(mfcc, n_frames, aNCep, aCepLifter);

  // Append energies
  if (aAppendEnergy) {
    for (i = 0; i < n_frames; i++) {
      mfcc[i][0] = logf(energy[i]);
    }
  }

  // Free unused arrays
  for (i = 0; i < n_frames; i++) {
    free(feat[i]);
  }
  free(feat);
  free(energy);

  // Return MFCC features
  *aMFCC = mfcc;

  return n_frames;
}

int
csf_fbank(const short* aSignal, unsigned int aSignalLen, int aSampleRate,
          float aWinLen, float aWinStep, int aNFilters, int aNFFT,
          int aLowFreq, int aHighFreq, float aPreemph,
          float*** aFeatures, float** aEnergy)
{
  int i, j, k;
  float** feat;
  float** fbank;
  float** pspec;
  float** frames;
  float* energy;
  float* preemph = csf_preemphasis(aSignal, aSignalLen, aPreemph);
  int frame_len = (int)roundf(aWinLen * aSampleRate);
  int frame_step = (int)roundf(aWinStep * aSampleRate);

  // Frame the signal into overlapping frames
  int n_frames = csf_framesig(preemph, aSignalLen, frame_len, aNFFT,
                              frame_step, &frames);

  // Free preemphasised signal buffer
  free(preemph);

  // Compute the power spectrum of the frames
  pspec = csf_powspec((const float**)frames, n_frames, aNFFT);

  // Free frames
  for (i = 0; i < n_frames; i++) {
    free(frames[i]);
  }
  free(frames);

  // Store the total energy in each frame
  energy = (float*)calloc(sizeof(float), n_frames);
  for (i = 0; i < n_frames; i++) {
    for (j = 0; j < aNFFT / 2 + 1; j++) {
      energy[i] += pspec[i][j];
    }
    if (energy[i] == 0.0f) {
      energy[i] = FLT_MIN;
    }
  }

  // Compute the filter-bank energies
  fbank = csf_get_filterbanks(aNFilters, aNFFT, aSampleRate,
                              aLowFreq, aHighFreq);
  feat = (float**)malloc(sizeof(float*) * n_frames);
  for (i = 0; i < n_frames; i++) {
    feat[i] = (float*)calloc(sizeof(float), aNFilters);
    for (j = 0; j < aNFilters; j++) {
      for (k = 0; k < aNFFT / 2 + 1; k++) {
        feat[i][j] += pspec[i][k] * fbank[j][k];
      }
      if (feat[i][j] == 0.0f) {
        feat[i][j] = FLT_MIN;
      }
    }
  }

  // Free fbank
  for (i = 0; i < aNFilters; i++) {
    free(fbank[i]);
  }
  free(fbank);

  // Free pspec
  for (i = 0; i < n_frames; i++) {
    free(pspec[i]);
  }
  free(pspec);

  // Return features and energies
  *aFeatures = feat;
  *aEnergy = energy;

  return n_frames;
}

void
csf_lifter(float** aCepstra, int aNFrames, int aNCep, int aCepLifter)
{
  for (int i = 0; i < aNFrames; i++) {
    for (int j = 0; j < aNCep; j++) {
      aCepstra[i][j] *= 1 + (aCepLifter / 2.0f) * sinf(M_PI * j / aCepLifter);
    }
  }
}

float**
csf_get_filterbanks(int aNFilters, int aNFFT, int aSampleRate,
                    int aLowFreq, int aHighFreq)
{
  int i, j;
  float lowmel = CSF_HZ2MEL(aLowFreq);
  float highmel = CSF_HZ2MEL(aHighFreq);
  int* bin = (int*)malloc(sizeof(int) * (aNFilters + 2));
  float** fbank = (float**)malloc(sizeof(float*) * aNFilters);

  for (i = 0; i < aNFilters + 2; i++) {
    float melpoint = ((highmel - lowmel) / (float)(aNFilters + 1) * i) + lowmel;
    bin[i] = (int)floorf((aNFFT + 1) *
                         CSF_MEL2HZ(melpoint) / (float)aSampleRate);
  }

  for (i = 0; i < aNFilters; i++) {
    int start = MIN(bin[i], bin[i+1]);
    int end = MAX(bin[i], bin[i+1]);
    fbank[i] = (float*)calloc(sizeof(float), aNFFT / 2 + 1);
    for (j = start; j < end; j++) {
      fbank[i][j] = (j - bin[i]) / (float)(bin[i+1]-bin[i]);
    }
    start = MIN(bin[i+1], bin[i+2]);
    end = MAX(bin[i+1], bin[i+2]);
    for (j = start; j < end; j++) {
      fbank[i][j] = (bin[i+2]-j) / (float)(bin[i+2]-bin[i+1]);
    }
  }
  free(bin);

  return fbank;
}

int
csf_framesig(const float* aSignal, unsigned int aSignalLen, int aFrameLen,
             int aPaddedFrameLen, int aFrameStep, float*** aFrames)
{
  int** indices;
  float** frames;
  int i, j, n_frames = 1;

  if (aSignalLen > aFrameLen) {
    n_frames = 1 + (int)ceilf((aSignalLen - aFrameLen) / (float)aFrameStep);
  }

  indices = (int**)malloc(sizeof(int*) * n_frames);
  for (i = 0; i < n_frames; i++) {
    int base = i * aFrameStep;
    indices[i] = (int*)malloc(sizeof(int) * aFrameLen);
    for (j = 0; j < aFrameLen; j++) {
      indices[i][j] = base + j;
    }
  }

  frames = (float**)malloc(sizeof(float*) * n_frames);
  for (i = 0; i < n_frames; i++) {
    frames[i] = (float*)malloc(sizeof(float) * MAX(aPaddedFrameLen, aFrameLen));
    for (j = 0; j < aFrameLen; j++) {
      int index = indices[i][j];
      frames[i][j] = index < aSignalLen ? aSignal[index] : 0.0f;
    }
    for (j = aFrameLen; j < aPaddedFrameLen; j++) {
      frames[i][j] = 0.0f;
    }
    free (indices[i]);
  }
  free(indices);

  *aFrames = frames;
  return n_frames;
}

float*
csf_preemphasis(const short* aSignal, unsigned int aSignalLen, float aCoeff)
{
  int i;
  float* preemph = (float*)malloc(sizeof(float) * aSignalLen);

  for (i = aSignalLen - 1; i >= 1; i--) {
    preemph[i] = aSignal[i] - aSignal[i-1] * aCoeff;
  }
  preemph[0] = (float)aSignal[0];

  return preemph;
}

float**
csf_magspec(const float** aFrames, int aNFrames, int aNFFT)
{
  int i, j;
  const int fft_out = aNFFT / 2 + 1;
  kiss_fftr_cfg cfg = kiss_fftr_alloc(aNFFT, 0, NULL, NULL);
  float** mspec = (float**)malloc(sizeof(float*) * aNFrames);
  kiss_fft_cpx* out = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * fft_out);

  for (i = 0; i < aNFrames; i++) {
    mspec[i] = (float*)malloc(sizeof(float) * fft_out);

    // Compute the magnitude spectrum
    kiss_fftr(cfg, aFrames[i], out);
    for (j = 0; j < fft_out; j++) {
      mspec[i][j] = sqrtf(pow(out[j].r, 2.0f) + pow(out[j].i, 2.0f));
    }
  }

  free(out);
  return mspec;
}

float**
csf_powspec(const float** aFrames, int aNFrames, int aNFFT)
{
  const int fft_out = aNFFT / 2 + 1;
  float** pspec = csf_magspec(aFrames, aNFrames, aNFFT);

  for (int i = 0; i < aNFrames; i++) {
    for (int j = 0; j < fft_out; j++) {
      // Compute the power spectrum
      pspec[i][j] = (1.0/aNFFT) * powf(pspec[i][j], 2.0f);
    }
  }
  return pspec;
}

