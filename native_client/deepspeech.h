#ifndef DEEPSPEECH_H
#define DEEPSPEECH_H

#include <cstddef>

#define DEEPSPEECH_EXPORT __attribute__ ((visibility("default")))

namespace DeepSpeech
{

  class Private;

  class StreamingState;

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
       * @param aAlphabetConfigPath The path to the configuration file specifying
       *                            the alphabet used by the network. See alphabet.h.
       * @param aBeamWidth The beam width used by the decoder. A larger beam
       *                   width generates better results at the cost of decoding
       *                   time.
       */
      Model(const char* aModelPath, int aNCep, int aNContext,
            const char* aAlphabetConfigPath, int aBeamWidth);

      /**
       * @brief Frees associated resources and destroys model object.
       */
      virtual ~Model();

      /**
       * @brief Enable decoding using beam scoring with a KenLM language model.
       *
       * @param aAlphabetConfigPath The path to the configuration file specifying
       *                            the alphabet used by the network. See alphabet.h.
       * @param aLMPath The path to the language model binary file.
       * @param aTriePath The path to the trie file build from the same vocabu-
       *                  lary as the language model binary.
       * @param aLMWeight The weight to give to language model results when sco-
       *                  ring.
       * @param aWordCountWeight The weight (penalty) to give to beams when in-
       *                         creasing the word count of the decoding.
       * @param aValidWordCountWeight The weight (bonus) to give to beams when
       *                              adding a new valid word to the decoding.
       */
      void enableDecoderWithLM(const char* aAlphabetConfigPath,
                               const char* aLMPath, const char* aTriePath,
                               float aLMWeight,
                               float aWordCountWeight,
                               float aValidWordCountWeight);

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
       * @param[out] aMfcc An array containing features, of shape
       *                   (@p aNFrames, ncep * ncontext). The user is
       *                   responsible for freeing the array.
       * @param[out] aNFrames (optional) The number of frames in @p aMfcc.
       * @param[out] aFrameLen (optional) The length of each frame
       *                       (ncep * ncontext) in @p aMfcc.
       */
      void getInputVector(const short* aBuffer,
                          unsigned int aBufferSize,
                          int aSampleRate,
                          float** aMfcc,
                          int* aNFrames = NULL,
                          int* aFrameLen = NULL);

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
      const char* stt(const short* aBuffer,
                      unsigned int aBufferSize,
                      int aSampleRate);

      /**
       * @brief Setup a context used for performing streaming inference.
       *        the context pointer returned by this function can then be passed
       *        to {@link feedAudioContent()} and {@link finishStream()}.
       *
       * @param aPreAllocFrames Number of timestep frames to reserve. One timestep
       *                        is equivalent to two window lengths (50ms), so
       *                        by default we reserve enough frames for 3 seconds
       *                        of audio.
       * @param aSampleRate The sample-rate of the audio signal.
       *
       * @return A context pointer that represents the streaming state. Can be
       *         null if an error occurs.
       */
      StreamingState* setupStream(unsigned int aPreAllocFrames = 150,
                                  unsigned int aSampleRate = 16000);

      /**
       * @brief Feed audio samples to an ongoing streaming inference.
       *
       * @param aCtx A streaming context pointer returned by {@link setupStream()}.
       * @param aBuffer An array of 16-bit, mono raw audio samples at the
       *                appropriate sample rate.
       * @param aBufferSize The number of samples in @p aBuffer.
       */
      void feedAudioContent(StreamingState* aCtx, const short* aBuffer, unsigned int aBufferSize);

      /**
       * @brief Signal the end of an audio signal to an ongoing streaming
       *        inference, returns the STT result over the whole audio signal.
       *
       * @param aCtx A streaming context pointer returned by {@link setupStream()}.
       *
       * @return The STT result. The user is responsible for freeing the string.
       *
       * @note This method will free the context pointer (@p aCtx).
       */
      const char* finishStream(StreamingState* aCtx);
  };

  extern "C"
  void print_versions();

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
   * @param[out] aMfcc An array containing features, of shape
   *                   (@p aNFrames, ncep * ncontext). The user is responsible
   *                   for freeing the array.
   * @param[out] aNFrames (optional) The number of frames in @p aMfcc.
   * @param[out] aFrameLen (optional) The length of each frame
   *                       (ncep * ncontext) in @p aMfcc.
   */
  extern "C"
  void audioToInputVector(const short* aBuffer,
                          unsigned int aBufferSize,
                          int aSampleRate,
                          int aNCep,
                          int aNContext,
                          float** aMfcc,
                          int* aNFrames = NULL,
                          int* aFrameLen = NULL);

}

#endif /* DEEPSPEECH_H */
