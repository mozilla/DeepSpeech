
#ifndef __DEEPSPEECH_H__
#define __DEEPSPEECH_H__

#include <cstddef>

namespace DeepSpeech
{

  class Private;

  class Model {
    private:
      Private* mPriv;

    public:
      /**
       * @brief An object providing an interface to a trained DeepSpeech model.
       *
       * @param aModelPath The path to the frozen model graph.
       * @param aNCep The number of cepstrum the model was trained with.
       * @param aNContext The context window the model was trained with.
       */
      Model(const char* aModelPath, int aNCep, int aNContext);

      /**
       * @brief Frees associated resources and destroys model object.
       */
      ~Model();

      /**
       * @brief Given audio, return a vector suitable for input to the
       *        DeepSpeech model.
       *
       * Extracts MFCC features from a given audio signal and adds the
       * appropriate amount of context to run inference on the DeepSpeech model.
       * This is equivalent to calling audioToInputVector() with the model's
       * cepstrum and context window.
       *
       * @param aBuffer A 16-bit, mono raw audio signal at the appropriate
       *                sample rate.
       * @param aBufferSize The sample-length of the audio signal.
       * @param aSampleRate The sample-rate of the audio signal.
       * @param[out] aMFCC An array containing features, of shape
       *                   (@p aNFrames, ncep * ncontext). The user is
       *                   responsible for freeing the array.
       * @param[out] aNFrames (optional) The number of frames in @p aMFCC.
       * @param[out] aFrameLen (optional) The length of each frame
       *                       (ncep * ncontext) in @p aMFCC.
       */
      void getInputVector(const short* aBuffer,
                          unsigned int aBufferSize,
                          int aSampleRate,
                          float** aMfcc,
                          int* aNFrames = NULL,
                          int* aFrameLen = NULL);

      /**
       * @brief Run inference on the given audio.
       *
       * Runs inference on the given input vector with the model.
       * See getInputVector().
       *
       * @param aMfcc MFCC features with the appropriate amount of context per
       *              frame.
       * @param aNFrames The number of frames in @p aMfcc.
       * @param aFrameLen (optional) The length of each frame in @p aMfcc. If
       *                  specified, this will be used to verify the array is
       *                  large enough.
       *
       * @return The resulting string after running inference. The user is
       *         responsible for freeing this string.
       */
      char* infer(float* aMfcc,
                  int aNFrames,
                  int aFrameLen = 0);

      /**
       * @brief Use the DeepSpeech model to perform Speech-To-Text.
       *
       * @param aBuffer A 16-bit, mono raw audio signal at the appropriate
       *                sample rate.
       * @param aBufferSize The number of samples in the audio signal.
       * @param aSampleRate The sample-rate of the audio signal.
       *
       * @return The STT result. The user is responsible for freeing the string.
       */
      char* stt(const short* aBuffer,
                unsigned int aBufferSize,
                int aSampleRate);
  };

}

#endif /* __DEEPSPEECH_H__ */
