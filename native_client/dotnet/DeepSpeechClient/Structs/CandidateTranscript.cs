using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient.Structs
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct CandidateTranscript
    {
        /// <summary>
        /// Native list of tokens.
        /// </summary>
        internal unsafe IntPtr tokens;
        /// <summary>
        /// Count of tokens from the native side.
        /// </summary>
        internal unsafe int num_tokens;
        /// <summary>
        /// Approximated confidence value for this transcription.
        /// </summary>
        internal unsafe double confidence;
    }
}
