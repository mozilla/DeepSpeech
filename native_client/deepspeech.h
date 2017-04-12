
#ifndef __DEEPSPEECH_H__
#define __DEEPSPEECH_H__

typedef struct _DeepSpeechContext DeepSpeechContext;

/**
 * @brief Initialise a DeepSpeech context.
 *
 * @param aModelPath The path to the frozen model graph.
 * @param aNCep The number of cepstrum the model was trained with.
 * @param aNContext The context window the model was trained with.
 *
 * @return A DeepSpeech context.
 */
DeepSpeechContext* DsInit(const char* aModelPath, int aNCep, int aNContext);

/**
 * @brief De-initialise a DeepSpeech context.
 *
 * @param aCtx A DeepSpeech context.
 */
void DsClose(DeepSpeechContext* aCtx);

/**
 * @brief Extract MFCC features from a given audio signal and add context.
 *
 * Extracts MFCC features from a given audio signal and adds the appropriate
 * amount of context to run inference with the given DeepSpeech context.
 *
 * @param aCtx A DeepSpeech context.
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate sample
 *                rate.
 * @param aBufferSize The sample-length of the audio signal.
 * @param aSampleRate The sample-rate of the audio signal.
 * @param[out] aMFCC An array containing features, of shape
 *                   (frames, ncep * ncontext). The user is responsible for
 *                   freeing the array.
 *
 * @return The number of frames in @p aMFCC.
 */
int DsGetMfccFrames(DeepSpeechContext* aCtx, const short* aBuffer,
                    size_t aBufferSize, int aSampleRate, float** aMfcc);

/**
 * @brief Run inference on the given audio.
 *
 * Runs inference on the given MFCC audio features with the given DeepSpeech
 * context. See DsGetMfccFrames().
 *
 * @param aCtx A DeepSpeech context.
 * @param aMfcc MFCC features with the appropriate amount of context per frame.
 * @param aNFrames The number of frames in @p aMfcc.
 *
 * @return The resulting string after running inference. The user is
 *         responsible for freeing this string.
 */
char* DsInfer(DeepSpeechContext* aCtx, float* aMfcc, int aNFrames);

/**
 * @brief Use DeepSpeech to perform Speech-To-Text.
 *
 * @param aMfcc An MFCC features array.
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate sample
 *                rate.
 * @param aBufferSize The number of samples in the audio signal.
 * @param aSampleRate The sample-rate of the audio signal.
 *
 * @return The STT result. The user is responsible for freeing this string.
 */
char* DsSTT(DeepSpeechContext* aCtx, const short* aBuffer, size_t aBufferSize,
            int aSampleRate);

#endif /* __DEEPSPEECH_H__ */
