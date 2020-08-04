using MozillaVoiceSttClient.Models;
using System;
using System.IO;

namespace MozillaVoiceSttClient.Interfaces
{
    /// <summary>
    /// Client interface of Mozilla Voice STT.
    /// </summary>
    public interface IModel : IDisposable
    {
        /// <summary>
        /// Return version of this library. The returned version is a semantic version
        /// (SemVer 2.0.0).
        /// </summary>
        unsafe string Version();

        /// <summary>
        /// Return the sample rate expected by the model.
        /// </summary>
        /// <returns>Sample rate.</returns>
        unsafe int GetModelSampleRate();

        /// <summary>
        /// Get beam width value used by the model. If SetModelBeamWidth was not
        /// called before, will return the default value loaded from the model
        /// file.
        /// </summary>
        /// <returns>Beam width value used by the model.</returns>
        unsafe uint GetModelBeamWidth();

        /// <summary>
        /// Set beam width value used by the model.
        /// </summary>
        /// <param name="aBeamWidth">The beam width used by the decoder. A larger beam width value generates better results at the cost of decoding time.</param>
        /// <exception cref="ArgumentException">Thrown on failure.</exception>
        unsafe void SetModelBeamWidth(uint aBeamWidth);

        /// <summary>
        /// Enable decoding using an external scorer.
        /// </summary>
        /// <param name="aScorerPath">The path to the external scorer file.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to enable decoding with an external scorer.</exception>
        /// <exception cref="FileNotFoundException">Thrown when cannot find the scorer file.</exception>
        unsafe void EnableExternalScorer(string aScorerPath);

        /// <summary>
        /// Disable decoding using an external scorer.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when an external scorer is not enabled.</exception>
        unsafe void DisableExternalScorer();

        /// <summary>
        /// Set hyperparameters alpha and beta of the external scorer.
        /// </summary>
        /// <param name="aAlpha">The alpha hyperparameter of the decoder. Language model weight.</param>
        /// <param name="aBeta">The beta hyperparameter of the decoder. Word insertion weight.</param>
        /// <exception cref="ArgumentException">Thrown when an external scorer is not enabled.</exception>
        unsafe void SetScorerAlphaBeta(float aAlpha, float aBeta);

        /// <summary>
        /// Use the Mozilla Voice STT model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The STT result. Returns NULL on error.</returns>
        unsafe string SpeechToText(short[] aBuffer,
                uint aBufferSize);

        /// <summary>
        /// Use the Mozilla Voice STT model to perform Speech-To-Text, return results including metadata.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata. Returns NULL on error.</returns>
        unsafe Metadata SpeechToTextWithMetadata(short[] aBuffer,
                uint aBufferSize,
                uint aNumResults);

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits.
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        unsafe void FreeStream(MozillaVoiceSttStream stream);

        /// <summary>
        /// Creates a new streaming inference state.
        /// </summary>
        unsafe MozillaVoiceSttStream CreateStream();

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to feed the data.</param>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).</param>
        unsafe void FeedAudioContent(MozillaVoiceSttStream stream, short[] aBuffer, uint aBufferSize);

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <returns>The STT intermediate result.</returns>
        unsafe string IntermediateDecode(MozillaVoiceSttStream stream);

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference, including metadata.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata result.</returns>
        unsafe Metadata IntermediateDecodeWithMetadata(MozillaVoiceSttStream stream, uint aNumResults);

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <returns>The STT result.</returns>
        unsafe string FinishStream(MozillaVoiceSttStream stream);

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal, including metadata.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata result.</returns>
        unsafe Metadata FinishStreamWithMetadata(MozillaVoiceSttStream stream, uint aNumResults);
    }
}
