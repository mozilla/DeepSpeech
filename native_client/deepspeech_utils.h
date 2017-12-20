#ifndef __DEEPSPEECH_UTILS_H__
#define __DEEPSPEECH_UTILS_H__

#include <cstddef>

#define DEEPSPEECH_EXPORT __attribute__ ((visibility("default")))

namespace DeepSpeech
{

  /**
   * @brief Given audio, return a vector suitable for input to a DeepSpeech
   *        model trained with the given parameters.
   *
   * Extracts MFCC features from a given audio signal and adds the appropriate
   * amount of context to run inference on a DeepSpeech model trained with
   * the given parameters.
   *
   * @param aBuffer A 16-bit, mono raw audio signal at the appropriate sample
   *                rate.
   * @param aBufferSize The sample-length of the audio signal.
   * @param aSampleRate The sample-rate of the audio signal.
   * @param aNCep The number of cepstrum.
   * @param aNContext The size of the context window.
   * @param[out] aMFCC An array containing features, of shape
   *                   (@p aNFrames, ncep * ncontext). The user is responsible
   *                   for freeing the array.
   * @param[out] aNFrames (optional) The number of frames in @p aMFCC.
   * @param[out] aFrameLen (optional) The length of each frame
   *                       (ncep * ncontext) in @p aMFCC.
   */
  void audioToInputVector(const short* aBuffer,
                          unsigned int aBufferSize,
                          int aSampleRate,
                          int aNCep,
                          int aNContext,
                          float** aMfcc,
                          int* aNFrames = NULL,
                          int* aFrameLen = NULL);

}

#endif /* __DEEPSPEECH_UTILS_H__ */
