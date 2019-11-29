using DeepSpeechClient.Models;
using System;
using System.IO;

namespace DeepSpeechClient.Interfaces
{
    /// <summary>
    /// Client interface of the Mozilla's DeepSpeech implementation.
    /// </summary>
    public interface IDeepSpeech : IDisposable
    {
        /// <summary>
        /// Prints the versions of Tensorflow and DeepSpeech.
        /// </summary>
        void PrintVersions();

        /// <summary>
        /// Return the sample rate expected by the model.
        /// </summary>
        /// <returns>Sample rate.</returns>
        unsafe int GetModelSampleRate();

        /// <summary>
        /// Enable decoding using beam scoring with a KenLM language model.
        /// </summary>
        /// <param name="aLMPath">The path to the language model binary file.</param>
        /// <param name="aTriePath">The path to the trie file build from the same vocabulary as the language model binary.</param>
        /// <param name="aLMAlpha">The alpha hyperparameter of the CTC decoder. Language Model weight.</param>
        /// <param name="aLMBeta">The beta hyperparameter of the CTC decoder. Word insertion weight.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to enable decoding with a language model.</exception>
        /// <exception cref="FileNotFoundException">Thrown when cannot find the language model or trie file.</exception>
        unsafe void EnableDecoderWithLM(string aLMPath,
                  string aTriePath,
                  float aLMAlpha,
                  float aLMBeta);

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The STT result. Returns NULL on error.</returns>
        unsafe string SpeechToText(short[] aBuffer,
                uint aBufferSize);

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The extended metadata. Returns NULL on error.</returns>
        unsafe Metadata SpeechToTextWithMetadata(short[] aBuffer,
                uint aBufferSize);

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits.
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        unsafe void FreeStream(DeepSpeechStream stream);

        /// <summary>
        /// Creates a new streaming inference state.
        /// </summary>
        unsafe DeepSpeechStream CreateStream();

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to feed the data.</param>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).</param>
        unsafe void FeedAudioContent(DeepSpeechStream stream, short[] aBuffer, uint aBufferSize);

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <returns>The STT intermediate result.</returns>
        unsafe string IntermediateDecode(DeepSpeechStream stream);

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <returns>The STT result.</returns>
        unsafe string FinishStream(DeepSpeechStream stream);

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <returns>The extended metadata result.</returns>
        unsafe Metadata FinishStreamWithMetadata(DeepSpeechStream stream);
    }
}
