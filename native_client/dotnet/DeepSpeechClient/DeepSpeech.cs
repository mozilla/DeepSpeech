using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Extensions;

using System;
using System.IO;
using DeepSpeechClient.Enums;

namespace DeepSpeechClient
{
    /// <summary>
    /// Client of the Mozilla's deepspeech implementation.
    /// </summary>
    public class DeepSpeech : IDeepSpeech
    {
        private unsafe IntPtr** _modelStatePP;
        private unsafe IntPtr** _streamingStatePP;




        public DeepSpeech()
        {

        }

        #region IDeepSpeech

        /// <summary>
        /// Create an object providing an interface to a trained DeepSpeech model.
        /// </summary>
        /// <param name="aModelPath">The path to the frozen model graph.</param>
        /// <param name="aAlphabetConfigPath">The path to the configuration file specifying the alphabet used by the network.</param>
        /// <param name="aBeamWidth">The beam width used by the decoder. A larger beam width generates better results at the cost of decoding time.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to create the model.</exception>
        public unsafe void CreateModel(string aModelPath,
            string aAlphabetConfigPath, uint aBeamWidth)
        {
            string exceptionMessage = null;
            if (string.IsNullOrWhiteSpace(aModelPath))
            {
                exceptionMessage = "Model path cannot be empty.";
            }
            if (string.IsNullOrWhiteSpace(aAlphabetConfigPath))
            {
                exceptionMessage = "Alphabet path cannot be empty.";
            }
            if (!File.Exists(aModelPath))
            {
                exceptionMessage = $"Cannot find the model file: {aModelPath}";
            }
            if (!File.Exists(aAlphabetConfigPath))
            {
                exceptionMessage = $"Cannot find the alphabet file: {aAlphabetConfigPath}";
            }

            if (exceptionMessage != null)
            {
                throw new FileNotFoundException(exceptionMessage);
            }
            var resultCode = NativeImp.DS_CreateModel(aModelPath,
                            aAlphabetConfigPath,
                            aBeamWidth,
                            ref _modelStatePP);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Evaluate the result code and will raise an exception if necessary.
        /// </summary>
        /// <param name="resultCode">Native result code.</param>
        private void EvaluateResultCode(ErrorCodes resultCode)
        {
            switch (resultCode)
            {
                case ErrorCodes.DS_ERR_OK:
                    break;
                case ErrorCodes.DS_ERR_NO_MODEL:
                    throw new ArgumentException("Missing model information.");
                case ErrorCodes.DS_ERR_INVALID_ALPHABET:
                    throw new ArgumentException("Invalid alphabet file or invalid alphabet size.");
                case ErrorCodes.DS_ERR_INVALID_SHAPE:
                    throw new ArgumentException("Invalid model shape.");
                case ErrorCodes.DS_ERR_INVALID_LM:
                    throw new ArgumentException("Invalid language model file.");
                case ErrorCodes.DS_ERR_FAIL_INIT_MMAP:
                    throw new ArgumentException("Failed to initialize memory mapped model.");
                case ErrorCodes.DS_ERR_FAIL_INIT_SESS:
                    throw new ArgumentException("Failed to initialize the session.");
                case ErrorCodes.DS_ERR_FAIL_INTERPRETER:
                    throw new ArgumentException("Interpreter failed.");
                case ErrorCodes.DS_ERR_FAIL_RUN_SESS:
                    throw new ArgumentException("Failed to run the session.");
                case ErrorCodes.DS_ERR_FAIL_CREATE_STREAM:
                    throw new ArgumentException("Error creating the stream.");
                case ErrorCodes.DS_ERR_FAIL_READ_PROTOBUF:
                    throw new ArgumentException("Error reading the proto buffer model file.");
                case ErrorCodes.DS_ERR_FAIL_CREATE_SESS:
                    throw new ArgumentException("Error failed to create session.");
                case ErrorCodes.DS_ERR_MODEL_INCOMPATIBLE:
                    throw new ArgumentException("Error incompatible model.");
                default:
                    throw new ArgumentException("Unknown error, please make sure you are using the correct native binary.");
            }
        }

        /// <summary>
        /// Frees associated resources and destroys models objects.
        /// </summary>
        public unsafe void Dispose()
        {
            NativeImp.DS_FreeModel(_modelStatePP);
        }

        /// <summary>
        /// Enable decoding using beam scoring with a KenLM language model.
        /// </summary>
        /// <param name="aLMPath">The path to the language model binary file.</param>
        /// <param name="aTriePath">The path to the trie file build from the same vocabulary as the language model binary.</param>
        /// <param name="aLMAlpha">The alpha hyperparameter of the CTC decoder. Language Model weight.</param>
        /// <param name="aLMBeta">The beta hyperparameter of the CTC decoder. Word insertion weight.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to enable decoding with a language model.</exception>
        public unsafe void EnableDecoderWithLM(string aLMPath, string aTriePath,
            float aLMAlpha, float aLMBeta)
        {
            string exceptionMessage = null;
            if (string.IsNullOrWhiteSpace(aTriePath))
            {
                exceptionMessage = "Path to the trie file cannot be empty.";
            }
            if (!File.Exists(aTriePath))
            {
                exceptionMessage = $"Cannot find the trie file: {aTriePath}";
            }

            if (exceptionMessage != null)
            {
                throw new FileNotFoundException(exceptionMessage);
            }

            var resultCode = NativeImp.DS_EnableDecoderWithLM(_modelStatePP,
                            aLMPath,
                            aTriePath,
                            aLMAlpha,
                            aLMBeta);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).</param>
        public unsafe void FeedAudioContent(short[] aBuffer, uint aBufferSize)
        {
            NativeImp.DS_FeedAudioContent(_streamingStatePP, aBuffer, aBufferSize);
        }

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <returns>The STT result. The user is responsible for freeing the string.</returns>
        public unsafe string FinishStream()
        {
            return NativeImp.DS_FinishStream(_streamingStatePP).PtrToString();
        }

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <returns>The extended metadata. The user is responsible for freeing the struct.</returns>
        public unsafe Models.Metadata FinishStreamWithMetadata()
        {
            return NativeImp.DS_FinishStreamWithMetadata(_streamingStatePP).PtrToMetadata();
        }

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference. This is an expensive process as the decoder implementation isn't
        /// currently capable of streaming, so it always starts from the beginning of the audio.
        /// </summary>
        /// <returns>The STT intermediate result. The user is responsible for freeing the string.</returns>
        public unsafe string IntermediateDecode()
        {
            return NativeImp.DS_IntermediateDecode(_streamingStatePP);
        }

        /// <summary>
        /// Prints the versions of Tensorflow and DeepSpeech.
        /// </summary>
        public unsafe void PrintVersions()
        {
            NativeImp.DS_PrintVersions();
        }

        /// <summary>
        /// Creates a new streaming inference state.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to initialize the streaming mode.</exception>
        public unsafe void CreateStream()
        {
            var resultCode = NativeImp.DS_CreateStream(_modelStatePP, ref _streamingStatePP);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits.
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        public unsafe void FreeStream()
        {
            NativeImp.DS_FreeStream(ref _streamingStatePP);
        }

        /// <summary>
        /// Free a DeepSpeech allocated string
        /// </summary>
        public unsafe void FreeString(IntPtr intPtr)
        {
            NativeImp.DS_FreeString(intPtr);
        }

        /// <summary>
        /// Free a DeepSpeech allocated Metadata struct
        /// </summary>
        public unsafe void FreeMetadata(IntPtr intPtr)
        {
            NativeImp.DS_FreeMetadata(intPtr);
        }

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The STT result. The user is responsible for freeing the string.  Returns NULL on error.</returns>
        public unsafe string SpeechToText(short[] aBuffer, uint aBufferSize)
        {
            return NativeImp.DS_SpeechToText(_modelStatePP, aBuffer, aBufferSize).PtrToString();
        }

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The extended metadata. The user is responsible for freeing the struct.  Returns NULL on error.</returns>
        public unsafe Models.Metadata SpeechToTextWithMetadata(short[] aBuffer, uint aBufferSize)
        {
            return NativeImp.DS_SpeechToTextWithMetadata(_modelStatePP, aBuffer, aBufferSize).PtrToMetadata();
        }

        #endregion



    }
}
