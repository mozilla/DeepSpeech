#ifndef DEEPSPEECH_H
#define DEEPSPEECH_H

#ifndef SWIG
#define DEEPSPEECH_EXPORT __attribute__ ((visibility("default")))
#else
#define DEEPSPEECH_EXPORT
#endif

struct ModelState;

struct StreamingState;

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
 * @param[out] retval a ModelState pointer
 *
 * @return Zero on success, non-zero on failure.
 */
DEEPSPEECH_EXPORT
int DS_CreateModel(char* aModelPath,
                   int aNCep,
                   int aNContext,
                   char* aAlphabetConfigPath,
                   int aBeamWidth,
                   ModelState** retval);

/**
 * @brief Frees associated resources and destroys model object.
 */
DEEPSPEECH_EXPORT
void DS_DestroyModel(ModelState* ctx);

/**
 * @brief Enable decoding using beam scoring with a KenLM language model.
 *
 * @param aCtx The ModelState pointer for the model being changed.
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
DEEPSPEECH_EXPORT
void DS_EnableDecoderWithLM(ModelState* aCtx,
                            char* aAlphabetConfigPath,
                            char* aLMPath,
                            char* aTriePath,
                            float aLMWeight,
                            float aWordCountWeight,
                            float aValidWordCountWeight);

/**
 * @brief Use the DeepSpeech model to perform Speech-To-Text.
 *
 * @param aCtx The ModelState pointer for the model to use.
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate
 *                sample rate.
 * @param aBufferSize The number of samples in the audio signal.
 * @param aSampleRate The sample-rate of the audio signal.
 *
 * @return The STT result. The user is responsible for freeing the string.
 */
DEEPSPEECH_EXPORT
char* DS_SpeechToText(ModelState* aCtx,
                      short* aBuffer,
                      unsigned int aBufferSize,
                      int aSampleRate);

/**
 * @brief Setup a context used for performing streaming inference.
 *        the context pointer returned by this function can then be passed
 *        to {@link DS_FeedAudioContent()} and {@link DS_FinishStream()}.
 *
 * @param aPreAllocFrames Number of timestep frames to reserve. One timestep
 *                        is equivalent to two window lengths (50ms), so
 *                        by default we reserve enough frames for 3 seconds
 *                        of audio.
 * @param aSampleRate The sample-rate of the audio signal.
 * @param[out] retval a context pointer that represents the streaming state. Can
 *                    be null if an error occurs.
 *
 * @return Zero for success, non-zero on failure.
 */
DEEPSPEECH_EXPORT
int DS_SetupStream(ModelState* aCtx,
                   unsigned int aPreAllocFrames,
                   unsigned int aSampleRate,
                   StreamingState** retval);

/**
 * @brief Feed audio samples to an ongoing streaming inference.
 *
 * @param aCtx A streaming context pointer returned by {@link DS_SetupStream()}.
 * @param aBuffer An array of 16-bit, mono raw audio samples at the
 *                appropriate sample rate.
 * @param aBufferSize The number of samples in @p aBuffer.
 */
DEEPSPEECH_EXPORT
void DS_FeedAudioContent(StreamingState* aSctx,
                         short* aBuffer,
                         unsigned int aBufferSize);

/**
 * @brief Signal the end of an audio signal to an ongoing streaming
 *        inference, returns the STT result over the whole audio signal.
 *
 * @param aSctx A streaming context pointer returned by {@link DS_SetupStream()}.
 *
 * @return The STT result. The user is responsible for freeing the string.
 *
 * @note This method will free the context pointer (@p aCtx).
 */
DEEPSPEECH_EXPORT
char* DS_FinishStream(StreamingState* aSctx);

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
DEEPSPEECH_EXPORT
void DS_AudioToInputVector(short* aBuffer,
                           unsigned int aBufferSize,
                           int aSampleRate,
                           int aNCep,
                           int aNContext,
                           float** aMfcc,
                           int* aNFrames = NULL,
                           int* aFrameLen = NULL);

/**
 * @brief Print version of this library and of the linked TensorFlow library.
 */
DEEPSPEECH_EXPORT
void DS_PrintVersions();

#undef DEEPSPEECH_EXPORT

#endif /* DEEPSPEECH_H */
