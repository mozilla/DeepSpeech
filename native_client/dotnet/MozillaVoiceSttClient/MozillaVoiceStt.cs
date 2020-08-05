using MozillaVoiceSttClient.Interfaces;
using MozillaVoiceSttClient.Extensions;

using System;
using System.IO;
using MozillaVoiceSttClient.Enums;
using MozillaVoiceSttClient.Models;

namespace MozillaVoiceSttClient
{
    /// <summary>
    /// Concrete implementation of <see cref="MozillaVoiceStt.Interfaces.IMozillaVoiceSttModel"/>.
    /// </summary>
    public class MozillaVoiceSttModel : IMozillaVoiceSttModel
    {
        private unsafe IntPtr** _modelStatePP;
        
        /// <summary>
        /// Initializes a new instance of <see cref="MozillaVoiceSttModel"/> class and creates a new acoustic model.
        /// </summary>
        /// <param name="aModelPath">The path to the frozen model graph.</param>
        /// <exception cref="ArgumentException">Thrown when the native binary failed to create the model.</exception>
        public MozillaVoiceSttModel(string aModelPath)
        {
            CreateModel(aModelPath);
        }

        #region IMozillaVoiceSttModel

        /// <summary>
        /// Create an object providing an interface to a trained Mozilla Voice STT model.
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
            var resultCode = NativeImp.STT_CreateModel(aModelPath,
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
            return NativeImp.STT_GetModelBeamWidth(_modelStatePP);
        }

        /// <summary>
        /// Set beam width value used by the model.
        /// </summary>
        /// <param name="aBeamWidth">The beam width used by the decoder. A larger beam width value generates better results at the cost of decoding time.</param>
        /// <exception cref="ArgumentException">Thrown on failure.</exception>
        public unsafe void SetModelBeamWidth(uint aBeamWidth)
        {
            var resultCode = NativeImp.STT_SetModelBeamWidth(_modelStatePP, aBeamWidth);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Return the sample rate expected by the model.
        /// </summary>
        /// <returns>Sample rate.</returns>
        public unsafe int GetModelSampleRate()
        {
            return NativeImp.STT_GetModelSampleRate(_modelStatePP);
        }

        /// <summary>
        /// Evaluate the result code and will raise an exception if necessary.
        /// </summary>
        /// <param name="resultCode">Native result code.</param>
        private void EvaluateResultCode(ErrorCodes resultCode)
        {
            if (resultCode != ErrorCodes.STT_ERR_OK)
            {
                throw new ArgumentException(NativeImp.STT_ErrorCodeToErrorMessage((int)resultCode).PtrToString());
            }
        }

        /// <summary>
        /// Frees associated resources and destroys models objects.
        /// </summary>
        public unsafe void Dispose()
        {
            NativeImp.STT_FreeModel(_modelStatePP);
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

            var resultCode = NativeImp.STT_EnableExternalScorer(_modelStatePP, aScorerPath);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Disable decoding using an external scorer.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when an external scorer is not enabled.</exception>
        public unsafe void DisableExternalScorer()
        {
            var resultCode = NativeImp.STT_DisableExternalScorer(_modelStatePP);
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
            var resultCode = NativeImp.STT_SetScorerAlphaBeta(_modelStatePP,
                            aAlpha,
                            aBeta);
            EvaluateResultCode(resultCode);
        }

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to feed the data.</param>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).</param>
        public unsafe void FeedAudioContent(MozillaVoiceSttStream stream, short[] aBuffer, uint aBufferSize)
        {
            NativeImp.STT_FeedAudioContent(stream.GetNativePointer(), aBuffer, aBufferSize);
        }

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <returns>The STT result.</returns>
        public unsafe string FinishStream(MozillaVoiceSttStream stream)
        {
            return NativeImp.STT_FinishStream(stream.GetNativePointer()).PtrToString();
        }

        /// <summary>
        /// Closes the ongoing streaming inference, returns the STT result over the whole audio signal, including metadata.
        /// </summary>
        /// <param name="stream">Instance of the stream to finish.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata result.</returns>
        public unsafe Metadata FinishStreamWithMetadata(MozillaVoiceSttStream stream, uint aNumResults)
        {
            return NativeImp.STT_FinishStreamWithMetadata(stream.GetNativePointer(), aNumResults).PtrToMetadata();
        }

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <returns>The STT intermediate result.</returns>
        public unsafe string IntermediateDecode(MozillaVoiceSttStream stream)
        {
            return NativeImp.STT_IntermediateDecode(stream.GetNativePointer()).PtrToString();
        }

        /// <summary>
        /// Computes the intermediate decoding of an ongoing streaming inference, including metadata.
        /// </summary>
        /// <param name="stream">Instance of the stream to decode.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The STT intermediate result.</returns>
        public unsafe Metadata IntermediateDecodeWithMetadata(MozillaVoiceSttStream stream, uint aNumResults)
        {
            return NativeImp.STT_IntermediateDecodeWithMetadata(stream.GetNativePointer(), aNumResults).PtrToMetadata();
        }

        /// <summary>
        /// Return version of this library. The returned version is a semantic version
        /// (SemVer 2.0.0).
        /// </summary>
        public unsafe string Version()
        {
            return NativeImp.STT_Version().PtrToString();
        }

        /// <summary>
        /// Creates a new streaming inference state.
        /// </summary>
        public unsafe MozillaVoiceSttStream CreateStream()
        {
            IntPtr** streamingStatePointer = null;
            var resultCode = NativeImp.STT_CreateStream(_modelStatePP, ref streamingStatePointer);
            EvaluateResultCode(resultCode);
            return new MozillaVoiceSttStream(streamingStatePointer);
        }

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits.
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        public unsafe void FreeStream(MozillaVoiceSttStream stream)
        {
            NativeImp.STT_FreeStream(stream.GetNativePointer());
            stream.Dispose();
        }

        /// <summary>
        /// Use the Mozilla Voice STT model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <returns>The STT result. Returns NULL on error.</returns>
        public unsafe string SpeechToText(short[] aBuffer, uint aBufferSize)
        {
            return NativeImp.STT_SpeechToText(_modelStatePP, aBuffer, aBufferSize).PtrToString();
        }

        /// <summary>
        /// Use the Mozilla Voice STT model to perform Speech-To-Text, return results including metadata.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <param name="aNumResults">Maximum number of candidate transcripts to return. Returned list might be smaller than this.</param>
        /// <returns>The extended metadata. Returns NULL on error.</returns>
        public unsafe Metadata SpeechToTextWithMetadata(short[] aBuffer, uint aBufferSize, uint aNumResults)
        {
            return NativeImp.STT_SpeechToTextWithMetadata(_modelStatePP, aBuffer, aBufferSize, aNumResults).PtrToMetadata();
        }

        #endregion



    }
}
