#ifndef DEEPSPEECH_COMPAT_H
#define DEEPSPEECH_COMPAT_H

#include "deepspeech.h"

#warning This header is a convenience wrapper for compatibility with \
         the previous API, it has deprecated function names and arguments. \
         If possible, update your code instead of using this header.

/**
 * @brief An object providing an interface to a trained DeepSpeech model.
 *
 * @param aModelPath The path to the frozen model graph.
 * @param aNCep UNUSED, DEPRECATED.
 * @param aNContext UNUSED, DEPRECATED.
 * @param aAlphabetConfigPath The path to the configuration file specifying
 *                            the alphabet used by the network. See alphabet.h.
 * @param aBeamWidth The beam width used by the decoder. A larger beam
 *                   width generates better results at the cost of decoding
 *                   time.
 * @param[out] retval a ModelState pointer
 *
 * @return Zero on success, non-zero on failure.
 */
int DS_CreateModel(const char* aModelPath,
                   unsigned int /*aNCep*/,
                   unsigned int /*aNContext*/,
                   const char* aAlphabetConfigPath,
                   unsigned int aBeamWidth,
                   ModelState** retval)
{
  return DS_CreateModel(aModelPath, aAlphabetConfigPath, aBeamWidth, retval);
}

/**
 * @brief Frees associated resources and destroys model object.
 */
void DS_DestroyModel(ModelState* ctx)
{
  return DS_FreeModel(ctx);
}

/**
 * @brief Enable decoding using beam scoring with a KenLM language model.
 *
 * @param aCtx The ModelState pointer for the model being changed.
 * @param aAlphabetConfigPath UNUSED, DEPRECATED.
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
int DS_EnableDecoderWithLM(ModelState* aCtx,
                           const char* /*aAlphabetConfigPath*/,
                           const char* aLMPath,
                           const char* aTriePath,
                           float aLMAlpha,
                           float aLMBeta)
{
  return DS_EnableDecoderWithLM(aCtx, aLMPath, aTriePath, aLMAlpha, aLMBeta);
}

/**
 * @brief Create a new streaming inference state. The streaming state returned
 *        by this function can then be passed to {@link DS_FeedAudioContent()}
 *        and {@link DS_FinishStream()}.
 *
 * @param aCtx The ModelState pointer for the model to use.
 * @param aSampleRate UNUSED, DEPRECATED.
 * @param[out] retval an opaque pointer that represents the streaming state. Can
 *                    be NULL if an error occurs.
 *
 * @return Zero for success, non-zero on failure.
 */
int DS_SetupStream(ModelState* aCtx,
                   unsigned int /*aSampleRate*/,
                   StreamingState** retval)
{
  return DS_CreateStream(aCtx, retval);
}

/**
 * @brief Destroy a streaming state without decoding the computed logits. This
 *        can be used if you no longer need the result of an ongoing streaming
 *        inference and don't want to perform a costly decode operation.
 *
 * @param aSctx A streaming state pointer returned by {@link DS_CreateStream()}.
 *
 * @note This method will free the state pointer (@p aSctx).
 */
void DS_DiscardStream(StreamingState* aSctx)
{
  return DS_FreeStream(aSctx);
}

/**
 * @brief Use the DeepSpeech model to perform Speech-To-Text.
 *
 * @param aCtx The ModelState pointer for the model to use.
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate
 *                sample rate.
 * @param aBufferSize The number of samples in the audio signal.
 * @param aSampleRate UNUSED, DEPRECATED.
 *
 * @return The STT result. The user is responsible for freeing the string using
 *         {@link DS_FreeString()}. Returns NULL on error.
 */
char* DS_SpeechToText(ModelState* aCtx,
                      const short* aBuffer,
                      unsigned int aBufferSize,
                      unsigned int /*aSampleRate*/)
{
  return DS_SpeechToText(aCtx, aBuffer, aBufferSize);
}

/**
 * @brief Use the DeepSpeech model to perform Speech-To-Text and output metadata
 * about the results.
 *
 * @param aCtx The ModelState pointer for the model to use.
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate
 *                sample rate.
 * @param aBufferSize The number of samples in the audio signal.
 * @param aSampleRate UNUSED, DEPRECATED.
 *
 * @return Outputs a struct of individual letters along with their timing information.
 *         The user is responsible for freeing Metadata by calling {@link DS_FreeMetadata()}. Returns NULL on error.
 */
Metadata* DS_SpeechToTextWithMetadata(ModelState* aCtx,
                                      const short* aBuffer,
                                      unsigned int aBufferSize,
                                      unsigned int /*aSampleRate*/)
{
  return DS_SpeechToTextWithMetadata(aCtx, aBuffer, aBufferSize);
}

#endif /* DEEPSPEECH_COMPAT_H */
