using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Extensions;

using System;
using System.IO;
using DeepSpeechClient.Enums;
using DeepSpeechClient.Models;

namespace DeepSpeechClient
{
    /// <summary>
    /// Concrete implementation of <see cref="DeepSpeechClient.Interfaces.IDeepSpeech"/>.
    /// </summary>
    public class DeepSpeech : IDeepSpeech
    {
        private unsafe IntPtr** _modelStatePP;
        
        /// <summary>
        /// Initializes a new instance of <see cref="DeepSpeech"/> class and creates a new acoustic model.
        /// </summary>
        /// <param name="aModelPath">The path to the frozen model graph.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to create the model.</exception>
        public DeepSpeech(string aModelPath)
        {
            CreateModel(aModelPath);
        }

        #region IDeepSpeech

        /// <summary>
        /// Create an object providing an interface to a trained DeepSpeech model.
        /// </summary>
        /// <param name="aModelPath">The path to the frozen model graph.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to create the model.</exception>
        private unsafe void CreateModel(string aModelPath)
        {
            string exceptionMessage = null;
            if (string.IsNullOrWhiteSpace(aModelPath))
            {
                exceptionMessage = "Model path cannot be empty.";
            }
            if (!File.Exists(aModelPath))
            {
                exceptionMessage = $"Cannot find the model file: {aModelPath}";
            }

            if (exceptionMessage != null)
            {
                throw new FileNotFoundException(exceptionMessage);
            }
            var resultCode = NativeImp.DS_CreateModel(aModelPath,
                            ref _modelStatePP);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Get beam width value used by the model. If SetModelBeamWidth was not
        /// called before, will return the default value loaded from the model file.
        /// </summary>
        /// <returns>Beam width value used by the model.</returns>
        public unsafe uint GetModelBeamWidth()
        {
            return NativeImp.DS_GetModelBeamWidth(_modelStatePP);
        }

        /// <summary>
        /// Set beam width value used by the model.
        /// </summary>
        /// <param name="aBeamWidth">The beam width used by the decoder. A larger beam width value generates better results at the cost of decoding time.</param>
        /// <exception cref="ArgumentException">Thrown on failure.</exception>
        public unsafe void SetModelBeamWidth(uint aBeamWidth)
        {
            var resultCode = NativeImp.DS_SetModelBeamWidth(_modelStatePP, aBeamWidth);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Add a hot-word.
        /// 
        /// Words that don't occur in the scorer (e.g. proper nouns) or strings that contain spaces won't be taken into account.
        /// </summary>
        /// <param name="aWord">Some word</param>
        /// <param name="aBoost">Some boost. Positive value increases and negative reduces chance of a word occuring in a transcription. Excessive positive boost might lead to splitting up of letters of the word following the hot-word.</param>
        /// <exception cref="ArgumentException">Thrown on failure.</exception>
        public unsafe void AddHotWord(string aWord, float aBoost)
        {
            var resultCode = NativeImp.DS_AddHotWord(_modelStatePP, aWord, aBoost);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Erase entry for a hot-word.
        /// </summary>
        /// <param name="aWord">Some word</param>
        /// <exception cref="ArgumentException">Thrown on failure.</exception>
        public unsafe void EraseHotWord(string aWord)
        {
            var resultCode = NativeImp.DS_EraseHotWord(_modelStatePP, aWord);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Clear all hot-words.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown on failure.</exception>
        public unsafe void ClearHotWords()
        {
            var resultCode = NativeImp.DS_ClearHotWords(_modelStatePP);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Return the sample rate expected by the model.
        /// </summary>
        /// <returns>Sample rate.</returns>
        public unsafe int GetModelSampleRate()
        {
            return NativeImp.DS_GetModelSampleRate(_modelStatePP);
        }

        /// <summary>
        /// Evaluate the result code and will raise an exception if necessary.
        /// </summary>
        /// <param name="resultCode">Native result code.</param>
        private void EvaluateResultCode(ErrorCodes resultCode)
        {
            if (resultCode != ErrorCodes.DS_ERR_OK)
            {
                throw new ArgumentException(NativeImp.DS_ErrorCodeToErrorMessage((int)resultCode).PtrToString());
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
        /// Enable decoding using an external scorer.
        /// </summary>
        /// <param name="aScorerPath">The path to the external scorer file.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to enable decoding with an external scorer.</exception>
        /// <exception cref="FileNotFoundException">Thrown when cannot find the scorer file.</exception>
        public unsafe void EnableExternalScorer(string aScorerPath)
        {
            if (string.IsNullOrWhiteSpace(aScorerPath))
            {
                throw new FileNotFoundException("Path to the scorer file cannot be empty.");
            }
            if (!File.Exists(aScorerPath))
            {
                throw new FileNotFoundException($"Cannot find the scorer file: {aScorerPath}");
            }

            var resultCode = NativeImp.DS_EnableExternalScorer(_modelStatePP, aScorerPath);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Disable decoding using an external scorer.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when an external scorer is not enabled.</exception>
        public unsafe void DisableExternalScorer()
        {
            var resultCode = NativeImp.DS_DisableExternalScorer(_modelStatePP);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Set hyperparameters alpha and beta of the external scorer.
        /// </summary>
        /// <param name="aAlpha">The alpha hyperparameter of the decoder. Language model weight.</param>
        /// <param name="aBeta">The beta hyperparameter of the decoder. Word insertion weight.</param>
        /// <exception cref="ArgumentException">Thrown when an external scorer is not enabled.</exception>
        public unsafe void SetScorerAlphaBeta(float aAlpha, float aBeta)
        {
            var resultCode = NativeImp.DS_SetScorerAlphaBeta(_modelStatePP,
                            aAlpha,
                            aBeta);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to feed the data.</param>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).</param>
        public unsafe void FeedAudioContent(DeepSpeechStream stream, short[] aBuffer, uint aBufferSize)
        {
            NativeImp.DS_FeedAudioContent(stream.GetNativePointer(), aBuffer, aBufferSize);
        }

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <returns>The STT result.</returns>
        public unsafe string FinishStream(DeepSpeechStream stream)
        {
            return NativeImp.DS_FinishStream(stream.GetNativePointer()).PtrToString();
        }

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal, including metadata.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata result.</returns>
        public unsafe Metadata FinishStreamWithMetadata(DeepSpeechStream stream, uint aNumResults)
        {
            return NativeImp.DS_FinishStreamWithMetadata(stream.GetNativePointer(), aNumResults).PtrToMetadata();
        }

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <returns>The STT intermediate result.</returns>
        public unsafe string IntermediateDecode(DeepSpeechStream stream)
        {
            return NativeImp.DS_IntermediateDecode(stream.GetNativePointer()).PtrToString();
        }

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference, including metadata.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The STT intermediate result.</returns>
        public unsafe Metadata IntermediateDecodeWithMetadata(DeepSpeechStream stream, uint aNumResults)
        {
            return NativeImp.DS_IntermediateDecodeWithMetadata(stream.GetNativePointer(), aNumResults).PtrToMetadata();
        }

        /// <summary>
        /// Return version of this library. The returned version is a semantic version
        /// (SemVer 2.0.0).
        /// </summary>
        public unsafe string Version()
        {
            return NativeImp.DS_Version().PtrToString();
        }

        /// <summary>
        /// Creates a new streaming inference state.
        /// </summary>
        public unsafe DeepSpeechStream CreateStream()
        {
            IntPtr** streamingStatePointer = null;
            var resultCode = NativeImp.DS_CreateStream(_modelStatePP, ref streamingStatePointer);
            EvaluateResultCode(resultCode);
            return new DeepSpeechStream(streamingStatePointer);
        }

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits.
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        public unsafe void FreeStream(DeepSpeechStream stream)
        {
            NativeImp.DS_FreeStream(stream.GetNativePointer());
            stream.Dispose();
        }

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The STT result. Returns NULL on error.</returns>
        public unsafe string SpeechToText(short[] aBuffer, uint aBufferSize)
        {
            return NativeImp.DS_SpeechToText(_modelStatePP, aBuffer, aBufferSize).PtrToString();
        }

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text, return results including metadata.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata. Returns NULL on error.</returns>
        public unsafe Metadata SpeechToTextWithMetadata(short[] aBuffer, uint aBufferSize, uint aNumResults)
        {
            return NativeImp.DS_SpeechToTextWithMetadata(_modelStatePP, aBuffer, aBufferSize, aNumResults).PtrToMetadata();
        }

        #endregion



    }
}
