using DeepSpeechClient.Models;
using System;

namespace DeepSpeechClient.Interfaces
{
    /// <summary>
    /// Client interface of the Mozilla's deepspeech implementation.
    /// </summary>
    public interface IDeepSpeech : IDisposable
    {
        /// <summary>
        /// Prints the versions of Tensorflow and DeepSpeech.
        /// </summary>
        void PrintVersions();

        /// <summary>
        /// Create an object providing an interface to a trained DeepSpeech model.
        /// </summary>
        /// <param name="aModelPath">The path to the frozen model graph.</param>
        /// <param name="aBeamWidth">The beam width used by the decoder. A larger beam width generates better results at the cost of decoding time.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to create the model.</exception>
        unsafe void CreateModel(string aModelPath,
                   uint aBeamWidth);

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
        unsafe void EnableDecoderWithLM(string aLMPath,
                  string aTriePath,
                  float aLMAlpha,
                  float aLMBeta);

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The STT result. The user is responsible for freeing the string.  Returns NULL on error.</returns>
        unsafe string SpeechToText(short[] aBuffer,
                uint aBufferSize);

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The extended metadata result. The user is responsible for freeing the struct.  Returns NULL on error.</returns>
        unsafe Metadata SpeechToTextWithMetadata(short[] aBuffer,
                uint aBufferSize);

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits.
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        unsafe void FreeStream();


        /// <summary>
        /// Creates a new streaming inference state.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to initialize the streaming mode.</exception>
        unsafe void CreateStream();

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).</param>
        unsafe void FeedAudioContent(short[] aBuffer, uint aBufferSize);

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference. This is an expensive process as the decoder implementation isn't
        /// currently capable of streaming, so it always starts from the beginning of the audio.
        /// </summary>
        /// <returns>The STT intermediate result. The user is responsible for freeing the string.</returns>
        unsafe string IntermediateDecode();

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <returns>The STT result. The user is responsible for freeing the string.</returns>
        unsafe string FinishStream();

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <returns>The extended metadata result. The user is responsible for freeing the struct.</returns>
        unsafe Metadata FinishStreamWithMetadata();
    }
}
