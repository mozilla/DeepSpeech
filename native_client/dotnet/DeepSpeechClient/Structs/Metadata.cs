using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient.Structs
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct Metadata
    {
        /// <summary>
        /// Native list of items.
        /// </summary>
        internal unsafe IntPtr items;
        /// <summary>
        /// Count of items from the native side.
        /// </summary>
        internal unsafe int num_items;
        /// <summary>
        /// Approximated confidence value for this transcription.
        /// </summary>
        internal unsafe double confidence;
    }
}
