using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Structs;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace DeepSpeechClient
{
    /// <summary>
    /// Client of the Mozilla's deepspeech implementation.
    /// </summary>
    public class DeepSpeech : IDeepSpeech
    {
        private unsafe ModelState** _modelStatePP;
        private unsafe ModelState* _modelStateP;
        private unsafe StreamingState** _streamingStatePP;

        


        public DeepSpeech()
        {

        }

        #region IDeepSpeech

        /// <summary>
        /// Create an object providing an interface to a trained DeepSpeech model.
        /// </summary>
        /// <param name="aModelPath">The path to the frozen model graph.</param>
        /// <param name="aNCep">The number of cepstrum the model was trained with.</param>
        /// <param name="aNContext">The context window the model was trained with.</param>
        /// <param name="aAlphabetConfigPath">The path to the configuration file specifying the alphabet used by the network.</param>
        /// <param name="aBeamWidth">The beam width used by the decoder. A larger beam width generates better results at the cost of decoding time.</param>
        /// <returns>Zero on success, non-zero on failure.</returns>
        public unsafe int CreateModel(string aModelPath, uint aNCep,
            uint aNContext, string aAlphabetConfigPath, uint aBeamWidth)
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
            int result = NativeImp.DS_CreateModel(aModelPath,
                            aNCep,
                            aNContext,
                            aAlphabetConfigPath,
                            aBeamWidth,
                            ref _modelStatePP);
            _modelStateP = *_modelStatePP;
            return result;


        }

        /// <summary>
        /// Frees associated resources and destroys models objects.
        /// </summary>
        public unsafe void Dispose()
        {
            NativeImp.DS_DestroyModel(_modelStatePP);
        }

        /// <summary>
        /// Enable decoding using beam scoring with a KenLM language model.
        /// </summary>
        /// <param name="aAlphabetConfigPath">The path to the configuration file specifying the alphabet used by the network.</param>
        /// <param name="aLMPath">The path to the language model binary file.</param>
        /// <param name="aTriePath">The path to the trie file build from the same vocabulary as the language model binary.</param>
        /// <param name="aLMAlpha">The alpha hyperparameter of the CTC decoder. Language Model weight.</param>
        /// <param name="aLMBeta">The beta hyperparameter of the CTC decoder. Word insertion weight.</param>
        /// <returns>Zero on success, non-zero on failure (invalid arguments).</returns>
        public unsafe int EnableDecoderWithLM(string aAlphabetConfigPath,
            string aLMPath, string aTriePath,
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

            return NativeImp.DS_EnableDecoderWithLM(_modelStatePP,
                            aAlphabetConfigPath,
                            aLMPath,
                            aTriePath,
                            aLMAlpha,
                            aLMBeta);
        }

        /// <summary>
        /// Feeds audio samples to an ongoing streaming inference.
        /// </summary>
        /// <param name="aBuffer">An array of 16-bit, mono raw audio samples at the appropriate sample rate.</param> 
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
            return NativeImp.DS_FinishStream(_streamingStatePP);
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
        /// <param name="aPreAllocFrames">Number of timestep frames to reserve.
        /// One timestep is equivalent to two window lengths(20ms). 
        /// If set to 0 we reserve enough frames for 3 seconds of audio(150).</param>
        /// <param name="aSampleRate">The sample-rate of the audio signal</param>
        /// <returns>Zero for success, non-zero on failure</returns>
        public unsafe int SetupStream(uint aPreAllocFrames, uint aSampleRate)
        {
            return NativeImp.DS_SetupStream(_modelStatePP, aPreAllocFrames, aSampleRate, ref _streamingStatePP);
        }

        /// <summary>
        /// Destroy a streaming state without decoding the computed logits. 
        /// This can be used if you no longer need the result of an ongoing streaming
        /// inference and don't want to perform a costly decode operation.
        /// </summary>
        public unsafe void DiscardStream()
        {
            NativeImp.DS_DiscardStream(ref _streamingStatePP);
        }

        /// <summary>
        /// Use the DeepSpeech model to perform Speech-To-Text.
        /// </summary>
        /// <param name="aBuffer">A 16-bit, mono raw audio signal at the appropriate sample rate.</param>
        /// <param name="aBufferSize">The number of samples in the audio signal.</param>
        /// <param name="aSampleRate">The sample-rate of the audio signal.</param>
        /// <returns>The STT result. The user is responsible for freeing the string.  Returns NULL on error.</returns>
        public unsafe string SpeechToText(short[] aBuffer, uint aBufferSize, uint aSampleRate)
        {
            var res = NativeImp.DS_SpeechToText(_modelStatePP, aBuffer, aBufferSize, aSampleRate);
            
            int len = 0;
            while (Marshal.ReadByte(res, len) != 0) ++len;
            byte[] buffer = new byte[len];
            Marshal.Copy(res, buffer, 0, buffer.Length);
            return Encoding.UTF8.GetString(buffer);
        }

        #endregion


        
    }
}
