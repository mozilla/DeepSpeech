#ifndef DEEPSPEECH_H
#define DEEPSPEECH_H

#ifndef SWIG
    #if defined _MSC_VER
        #define DEEPSPEECH_EXPORT extern "C" __declspec(dllexport) 
    #else                                                                   /*End of _MSC_VER*/  
        #define DEEPSPEECH_EXPORT __attribute__ ((visibility("default")))
#endif                                                                      /*End of SWIG*/  
#else
    #define DEEPSPEECH_EXPORT
#endif

struct ModelState;

struct StreamingState;

// Stores each individual character, along with its timing information
struct MetadataItem {
  char* character;
  int timestep; // Position of the character in units of 20ms
  float start_time; // Position of the character in seconds
};

// Stores the entire CTC output as an array of character metadata objects
struct Metadata {
  MetadataItem* items;
  int num_items;
};

enum DeepSpeech_Error_Codes
{
    // OK
    DS_ERR_OK                 = 0x0000,

    // Missing invormations
    DS_ERR_NO_MODEL           = 0x1000,

    // Invalid parameters
    DS_ERR_INVALID_ALPHABET   = 0x2000,
    DS_ERR_INVALID_SHAPE      = 0x2001,
    DS_ERR_INVALID_LM         = 0x2002,

    // Runtime failures
    DS_ERR_FAIL_INIT_MMAP     = 0x3000,
    DS_ERR_FAIL_INIT_SESS     = 0x3001,
    DS_ERR_FAIL_INTERPRETER   = 0x3002,
    DS_ERR_FAIL_RUN_SESS      = 0x3003,
    DS_ERR_FAIL_CREATE_STREAM = 0x3004,
    DS_ERR_FAIL_READ_PROTOBUF = 0x3005,
    DS_ERR_FAIL_CREATE_SESS   = 0x3006,
};

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
int DS_CreateModel(const char* aModelPath,
                   unsigned int aNCep,
                   unsigned int aNContext,
                   const char* aAlphabetConfigPath,
                   unsigned int aBeamWidth,
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
 * @param aLMAlpha The alpha hyperparameter of the CTC decoder. Language Model
                   weight.
 * @param aLMBeta The beta hyperparameter of the CTC decoder. Word insertion
                  weight.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
DEEPSPEECH_EXPORT
int DS_EnableDecoderWithLM(ModelState* aCtx,
                           const char* aAlphabetConfigPath,
                           const char* aLMPath,
                           const char* aTriePath,
                           float aLMAlpha,
                           float aLMBeta);

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
 *         Returns NULL on error.
 */
DEEPSPEECH_EXPORT
char* DS_SpeechToText(ModelState* aCtx,
                      const short* aBuffer,
                      unsigned int aBufferSize,
                      unsigned int aSampleRate);

/**
 * @brief Use the DeepSpeech model to perform Speech-To-Text and output metadata 
 * about the results.
 *
 * @param aCtx The ModelState pointer for the model to use.
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate
 *                sample rate.
 * @param aBufferSize The number of samples in the audio signal.
 * @param aSampleRate The sample-rate of the audio signal.
 *
 * @return Outputs a struct of individual letters along with their timing information. 
 *         The user is responsible for freeing Metadata by calling {@link DS_FreeMetadata()}. Returns NULL on error.
 */
DEEPSPEECH_EXPORT
Metadata* DS_SpeechToTextWithMetadata(ModelState* aCtx,
                      const short* aBuffer,
                      unsigned int aBufferSize,
                      unsigned int aSampleRate);

/**
 * @brief Create a new streaming inference state. The streaming state returned
 *        by this function can then be passed to {@link DS_FeedAudioContent()}
 *        and {@link DS_FinishStream()}.
 *
 * @param aCtx The ModelState pointer for the model to use.
 * @param aPreAllocFrames Number of timestep frames to reserve. One timestep
 *                        is equivalent to two window lengths (20ms). If set to 
 *                        0 we reserve enough frames for 3 seconds of audio (150).
 * @param aSampleRate The sample-rate of the audio signal.
 * @param[out] retval an opaque pointer that represents the streaming state. Can
 *                    be NULL if an error occurs.
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
 * @param aSctx A streaming state pointer returned by {@link DS_SetupStream()}.
 * @param aBuffer An array of 16-bit, mono raw audio samples at the
 *                appropriate sample rate.
 * @param aBufferSize The number of samples in @p aBuffer.
 */
DEEPSPEECH_EXPORT
void DS_FeedAudioContent(StreamingState* aSctx,
                         const short* aBuffer,
                         unsigned int aBufferSize);

/**
 * @brief Compute the intermediate decoding of an ongoing streaming inference.
 *        This is an expensive process as the decoder implementation isn't
 *        currently capable of streaming, so it always starts from the beginning
 *        of the audio.
 *
 * @param aSctx A streaming state pointer returned by {@link DS_SetupStream()}.
 *
 * @return The STT intermediate result. The user is responsible for freeing the
 *         string.
 */
DEEPSPEECH_EXPORT
char* DS_IntermediateDecode(StreamingState* aSctx);

/**
 * @brief Signal the end of an audio signal to an ongoing streaming
 *        inference, returns the STT result over the whole audio signal.
 *
 * @param aSctx A streaming state pointer returned by {@link DS_SetupStream()}.
 *
 * @return The STT result. The user is responsible for freeing the string.
 *
 * @note This method will free the state pointer (@p aSctx).
 */
DEEPSPEECH_EXPORT
char* DS_FinishStream(StreamingState* aSctx);

/**
 * @brief Signal the end of an audio signal to an ongoing streaming
 *        inference, returns per-letter metadata.
 *
 * @param aSctx A streaming state pointer returned by {@link DS_SetupStream()}.
 *
 * @return Outputs a struct of individual letters along with their timing information. 
 *         The user is responsible for freeing Metadata by calling {@link DS_FreeMetadata()}. Returns NULL on error.
 *
 * @note This method will free the state pointer (@p aSctx).
 */
DEEPSPEECH_EXPORT
Metadata* DS_FinishStreamWithMetadata(StreamingState* aSctx);

/**
 * @brief Destroy a streaming state without decoding the computed logits. This
 *        can be used if you no longer need the result of an ongoing streaming
 *        inference and don't want to perform a costly decode operation.
 *
 * @param aSctx A streaming state pointer returned by {@link DS_SetupStream()}.
 *
 * @note This method will free the state pointer (@p aSctx).
 */
DEEPSPEECH_EXPORT
void DS_DiscardStream(StreamingState* aSctx);

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
void DS_AudioToInputVector(const short* aBuffer,
                           unsigned int aBufferSize,
                           unsigned int aSampleRate,
                           unsigned int aNCep,
                           unsigned int aNContext,
                           float** aMfcc,
                           int* aNFrames = NULL,
                           int* aFrameLen = NULL);

/**
 * @brief Free memory allocated for metadata information.
 */

DEEPSPEECH_EXPORT
void DS_FreeMetadata(Metadata* m); 

/**
 * @brief Print version of this library and of the linked TensorFlow library.
 */
DEEPSPEECH_EXPORT
void DS_PrintVersions();

#undef DEEPSPEECH_EXPORT

#endif /* DEEPSPEECH_H */
