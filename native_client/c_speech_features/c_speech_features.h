
/**
 * Calculate filterbank features. Provides e.g. fbank and mfcc features for use
 * in ASR applications.
 *
 * Derived from python_speech_features, by James Lyons.
 * Port by Chris Lord.
 */

#ifndef __C_SPEECH_FEATURES_H__
#define __C_SPEECH_FEATURES_H__

#include <math.h>
#include "c_speech_features_config.h"

#define CSF_HZ2MEL(x) (2595.0 * csf_log10(1.0+(x)/700.0))
#define CSF_MEL2HZ(x) (700.0 * (csf_pow(10.0, (x)/2595.0) - 1.0))

#define CSF_2D_INDEX(w,x,y) (((y)*(w))+(x))
#define CSF_2D_REF(m,w,x,y) ((m)[CSF_2D_INDEX(w,x,y)])

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute MFCC features from an audio signal.
 *
 * @param aSignal The audio signal from which to compute features.
 * @param aSignalLen The length of the audio signal array.
 * @param aSampleRate The sample-rate of the signal.
 * @param aWinLen The length of the analysis window in seconds. (e.g. 0.025)
 * @param aWinStep The step between successive windows in seconds. (e.g. 0.01)
 * @param aNCep The number of cepstrum to return. (e.g. 13)
 * @param aNFilters The number of filters in the filterbank. (e.g. 26)
 * @param aNFFT The FFT size. (e.g. 512)
 * @param aLowFreq The lowest band edge of mel filters, in hz. (e.g. 0)
 * @param aHighFreq The highest band edge of mel filters, in hz. Must not be
 *                  higher than @p aSampleRate / 2. If this is lower or equal
 *                  to @p aLowFreq, it will be treated as @p aSampleRate / 2.
 * @param aPreemph Preemphasis filter coefficient. 0 is no filter. (e.g. 0.97)
 * @param aCepLifter The lifting coefficient to use. 0 disables lifting.
 *                   (e.g. 22)
 * @param aAppendEnergy If this is true, the zeroth cepstral coefficient is
 *                      replaced with the log of the total frame energy.
 * @param aWinFunc An array of size @c frameLen, as determined by multiplying
 *                 @p aWinLen by @p aSmapleRate, or @c NULL to be used as an
 *                 analysis window to apply to each frame. Refer to
 *                 csf_framesig().
 * @param[out] aMFCC An array containing features, of shape
 *                   (frames, @p aNCep). The user is responsible for freeing
 *                   the array.
 *
 * @return The number of frames.
 */
int csf_mfcc(const short* aSignal,
             unsigned int aSignalLen,
             int aSampleRate,
             csf_float aWinLen,
             csf_float aWinStep,
             int aNCep,
             int aNFilters,
             int aNFFT,
             int aLowFreq,
             int aHighFreq,
             csf_float aPreemph,
             int aCepLifter,
             int aAppendEnergy,
             csf_float* aWinFunc,
             csf_float** aMFCC);

/**
 * @brief Compute Mel-filterbank energy features from an audio signal.
 *
 * Compute Mel-filterbank energy features from an audio signal.
 *
 * @param aSignal The audio signal from which to compute features.
 * @param aSignalLen The length of the audio signal array.
 * @param aSampleRate The sample-rate of the signal.
 * @param aWinLen The length of the analysis window in seconds. (e.g. 0.025)
 * @param aWinStep The step between successive windows in seconds. (e.g. 0.01)
 * @param aNFilters The number of filters in the filterbank. (e.g. 26)
 * @param aNFFT The FFT size. (e.g. 512)
 * @param aLowFreq The lowest band edge of mel filters, in hz. (e.g. 0)
 * @param aHighFreq The highest band edge of mel filters, in hz. Must not be
 *                  higher than @p aSampleRate / 2. If this is lower or equal
 *                  to @p aLowFreq, it will be treated as @p aSampleRate / 2.
 * @param aPreemph Preemphasis filter coefficient. 0 is no filter. (e.g. 0.97)
 * @param aWinFunc An array of size @c frameLen, as determined by multiplying
 *                 @p aWinLen by @p aSmapleRate, or @c NULL to be used as an
 *                 analysis window to apply to each frame. Refer to
 *                 csf_framesig().
 * @param[out] aFeatures A 2D array containing features, of shape
 *                       (frames, @p aNFilters). The user is responsible for
 *                       freeing the array.
 * @param[out] aEnergy An array containing energies, of shape (frames), or
 *                     @c NULL. The user is responsible for freeing the array.
 *
 * @return The number of frames.
 */
int csf_fbank(const short* aSignal,
              unsigned int aSignalLen,
              int aSampleRate,
              csf_float aWinLen,
              csf_float aWinStep,
              int aNFilters,
              int aNFFT,
              int aLowFreq,
              int aHighFreq,
              csf_float aPreemph,
              csf_float* aWinFunc,
              csf_float** aFeatures,
              csf_float** aEnergy);

/**
 * @brief Compute log Mel-filterbank energy features from an audio signal.
 *
 * Compute log Mel-filterbank energy features from an audio signal.
 *
 * @param aSignal The audio signal from which to compute features.
 * @param aSignalLen The length of the audio signal array.
 * @param aSampleRate The sample-rate of the signal.
 * @param aWinLen The length of the analysis window in seconds. (e.g. 0.025)
 * @param aWinStep The step between successive windows in seconds. (e.g. 0.01)
 * @param aNFilters The number of filters in the filterbank. (e.g. 26)
 * @param aNFFT The FFT size. (e.g. 512)
 * @param aLowFreq The lowest band edge of mel filters, in hz. (e.g. 0)
 * @param aHighFreq The highest band edge of mel filters, in hz. Must not be
 *                  higher than @p aSampleRate / 2. If this is lower or equal
 *                  to @p aLowFreq, it will be treated as @p aSampleRate / 2.
 * @param aPreemph Preemphasis filter coefficient. 0 is no filter. (e.g. 0.97)
 * @param aWinFunc An array of size @c frameLen, as determined by multiplying
 *                 @p aWinLen by @p aSmapleRate, or @c NULL to be used as an
 *                 analysis window to apply to each frame. Refer to
 *                 csf_framesig().
 * @param[out] aFeatures A 2D array containing features, of shape
 *                       (frames, @p aNFilters). The user is responsible for
 *                       freeing the array.
 * @param[out] aEnergy An array containing energies, of shape (frames). The
 *                     user is responsible for freeing the array.
 *
 * @return The number of frames.
 */
int csf_logfbank(const short* aSignal,
                 unsigned int aSignalLen,
                 int aSampleRate,
                 csf_float aWinLen,
                 csf_float aWinStep,
                 int aNFilters,
                 int aNFFT,
                 int aLowFreq,
                 int aHighFreq,
                 csf_float aPreemph,
                 csf_float* aWinFunc,
                 csf_float** aFeatures,
                 csf_float** aEnergy);

/**
 * @brief Compute Spectral Sub-band Centroid features from an audio signal.
 *
 * Compute Spectral Sub-band Centroid features from an audio signal.
 *
 * @param aSignal The audio signal from which to compute features.
 * @param aSignalLen The length of the audio signal array.
 * @param aSampleRate The sample-rate of the signal.
 * @param aWinLen The length of the analysis window in seconds. (e.g. 0.025)
 * @param aWinStep The step between successive windows in seconds. (e.g. 0.01)
 * @param aNFilters The number of filters in the filterbank. (e.g. 26)
 * @param aNFFT The FFT size. (e.g. 512)
 * @param aLowFreq The lowest band edge of mel filters, in hz. (e.g. 0)
 * @param aHighFreq The highest band edge of mel filters, in hz. Must not be
 *                  higher than @p aSampleRate / 2. If this is lower or equal
 *                  to @p aLowFreq, it will be treated as @p aSampleRate / 2.
 * @param aPreemph Preemphasis filter coefficient. 0 is no filter. (e.g. 0.97)
 * @param aWinFunc An array of size @c frameLen, as determined by multiplying
 *                 @p aWinLen by @p aSmapleRate, or @c NULL to be used as an
 *                 analysis window to apply to each frame. Refer to
 *                 csf_framesig().
 * @param[out] aFeatures A 2D array containing features, of shape
 *                       (frames, @p aNFilters). The user is responsible for
 *                       freeing the array.
 */
int csf_ssc(const short* aSignal,
            unsigned int aSignalLen,
            int aSampleRate,
            csf_float aWinLen,
            csf_float aWinStep,
            int aNFilters,
            int aNFFT,
            int aLowFreq,
            int aHighFreq,
            csf_float aPreemph,
            csf_float* aWinFunc,
            csf_float** aFeatures);

/**
 * @brief Convert a value in Hertz to Mels
 *
 * Convert a value in Hertz to Mels
 *
 * @param aHz A value in Hz.
 *
 * @return A value in Mels.
 */
csf_float csf_hz2mel(csf_float aHz);

/**
 * @brief Convert a value in Mels to Hertz
 *
 * Convert a value in Mels to Hertz
 *
 * @param aMel A value in Mels.
 *
 * @return A value in Hz.
 */
csf_float csf_mel2hz(csf_float aMel);

/**
 * @brief Compute a Mel-filterbank.
 *
 * Compute a Mel-filterbank. The filters are stored in the rows, the columns
 * correspond to fft bins. The filters are returned as an array of size
 * @p aNFilters * (@p aNFFT / 2 + 1).
 *
 * @param aNFilters The number of filters in the filterbank. (e.g. 20)
 * @param aNFFT The FFT size. (e.g. 512)
 * @param aSampleRate The sample-rate of the signal being worked with. Affects
 *                    mel spacing.
 * @param aLowFreq The lowest band edge of mel filters, in hz. (e.g. 0)
 * @param aHighFreq The highest band edge of mel filters, in hz. Must not be
 *                  higher than @p aSampleRate / 2. If this is lower or equal
 *                  to @p aLowFreq, it will be treated as @p aSampleRate / 2.
 *
 * @return A 2D array of shape (@p aNFilters, @p aNFFT / 2 + 1). The user is
 *         responsible for freeing the array.
 */
csf_float* csf_get_filterbanks(int aNFilters,
                               int aNFFT,
                               int aSampleRate,
                               int aLowFreq,
                               int aHighFreq);

/**
 * @brief Apply a cepstral lifter on a matrix of cepstra.
 *
 * Apply a cepstral lifter on a matrix of cepstra. This has the effect of
 * increasing the magnitude of high-frequency DCT coefficients.
 *
 * @param aCepstra The 2D array matrix of mel-cepstra.
 * @param aNFrames The number of frames.
 * @param aNCep The number of cepstra per frame.
 * @param aCepLifter The lifting coefficient to use. 0 disables lifting.
 *                   (e.g. 22)
 */
void csf_lifter(csf_float* aCepstra,
                int aNFrames,
                int aNCep,
                int aCepLifter);

/**
 * @brief Compute delta features from a feature vector sequence.
 *
 * Compute delta features from a feature vector sequence.
 *
 * @param aFeatures A 2D array of shape (@p aNFeatures, @p aNFrames). Each row
 *                  holds one feature vector.
 * @param aNFrames The number of frames in @p aFeatures.
 * @param aNFrameLen The length of each frame in @p aFeatures.
 * @param @aN For each frame, calculate delta features based on preceding and
 *            following N frames. Must be 1 or larger.
 *
 * @return A 2D array of shape (@p aNFeatures, @p aNFrames) containing delta
 *         features. Each row contains holds 1 delta feature vector. The user
 *         is responsible for freeing the array.
 */
csf_float* csf_delta(const csf_float* aFeatures,
                     int aNFrames,
                     int aNFrameLen,
                     int aN);

/**
 * @brief Perform preemphasis on an input signal.
 *
 * Perform preemphasis on an input signal.
 *
 * @param aSignal The signal to filter.
 * @param aSignalLen The length of the signal array.
 * @param aCoeff The preemphasis coefficient. 0 is no filter. (e.g. 0.95)
 *
 * @return The filtered signal. The user is responsible for freeing this array.
 */
csf_float* csf_preemphasis(const short* aSignal,
                           unsigned int aSignalLen,
                           csf_float aCoeff);

/**
 * @brief Frame a signal into overlapping frames.
 *
 * Frame a signal into overlapping frames.
 *
 * @param aSignal The signal to frame.
 * @param aSignalLen The length of the signal array.
 * @param aFrameLen The length of each frame in samples.
 * @param aPaddedFrameLen If greater than @p aFrameLen, @p aPaddedFrameLen -
 *                        @p aFrameLen zeros will be appended to each frame.
 * @param aFrameStep The number of samples after the start of the previous frame
 *                   that the next frame should begin.
 * @param aWinFunc An array of size @p aFrameLen, or @c NULL to be used as an
 *                 analysis window to apply to each frame. When specified,
 *                 each overlapping frame of the signal will be multiplied
 *                 by the value in the corresponding index of the array.
 * @param[out] aFrames A 2D array of frames, of shape
 *                     (@c frames, @p aPaddedFrameLen).
 *                     The user is responsible for freeing the array.
 *
 * @return The number of frames.
 */
int csf_framesig(const csf_float* aSignal,
                 unsigned int aSignalLen,
                 int aFrameLen,
                 int aPaddedFrameLen,
                 int aFrameStep,
                 csf_float* aWinFunc,
                 csf_float** aFrames);

/**
 * @brief Perform overlap-add procedure to undo the action of csf_framesig().
 *
 * Perform overlap-add procedure to undo the action of csf_framesig().
 *
 * @param aFrames The 2D array of frames.
 * @param aNFrames The number of frames in @p aFrames.
 * @param aSigLen The length of the desired signal, or 0 if unknown.
 * @param aFrameLen The length of each frame in samples.
 * @param aFrameStep The number of samples after the start of the previous frame
 *                   that the next frame begins
 * @param aWinFunc An array of size @p aFrameLen, or @c NULL to be used as an
 *                 analysis window to apply to each frame. When specified,
 *                 each sample of the signal will be divided by the aggregated
 *                 value in the corresponding indices of the array.
 * @param[out] aSignal An array of samples. The length will be @p aSigLen if
 *                     specified. The user is responsible for freeing
 *                     this array.
 *
 * @return Returns the length of @p aSignal.
 */
int csf_deframesig(const csf_float* aFrames,
                   int aNFrames,
                   int aSigLen,
                   int aFrameLen,
                   int aFrameStep,
                   csf_float* aWinFunc,
                   csf_float** aSignal);

/**
 * @brief Compute the magnitude spectrum of frames.
 *
 * Compute the magnitude spectrum of each frame in frames.
 *
 * @param aFrames The 2D array of frames, of shape (@p aNFrames, @p aNFFT).
 * @param aNFrames The number of frames.
 * @param aNFFT The FFT length to use.
 *
 * @return A 2D array containing the magnitude spectrum of the
 *         corresponding frame, of shape (@p aNFrames, @p aNFFT / 2 + 1). The
 *         user is responsible for freeing the array.
 */
csf_float* csf_magspec(const csf_float* aFrames,
                       int aNFrames,
                       int aNFFT);

/**
 * @brief Compute the power spectrum of frames.
 *
 * Compute the power spectrum of each frame in frames.
 *
 * @param aFrames The 2D array of frames, of shape (@p aNFrames, @p aNFFT).
 * @param aNFrames The number of frames.
 * @param aNFFT The FFT length to use.
 *
 * @return A 2D array containing the power spectrum of the
 *         corresponding frame, of shape (@p aNFrames, @p aNFFT / 2 + 1).
 *         The user is responsible for freeing the array.
 */
csf_float* csf_powspec(const csf_float* aFrames,
                       int aNFrames,
                       int aNFFT);

/**
 * @brief Compute the log power spectrum of frames.
 *
 * Compute the log power spectrum of each frame in frames.
 *
 * @param aFrames The 2D array of frames, of shape (@p aNFrames, @p aNFFT).
 * @param aNFrames The number of frames.
 * @param aNFFT The FFT length to use.
 * @param aNorm If not zero, the log power spectrum is normalised so that the
 *              maximum value across all frames is 0.
 *
 * @return A 2D array containing the log power spectrum of the
 *         corresponding frame, of shape (@p aNFrames, @p aNFFT / 2 + 1).
 *         The user is responsible for freeing the array.
 */
csf_float* csf_logpowspec(const csf_float* aFrames,
                          int aNFrames,
                          int aNFFT,
                          int aNorm);

#ifdef __cplusplus
}
#endif

#endif /* __C_SPEECH_FEATURES_H__ */
