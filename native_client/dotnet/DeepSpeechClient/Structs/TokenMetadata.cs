using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient.Structs
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct TokenMetadata
    {
        /// <summary>
        /// Native text.
        /// </summary>
        internal unsafe IntPtr text;
        /// <summary>
        /// Position of the character in units of 20ms.
        /// </summary>
        internal unsafe int timestep;
        /// <summary>
        /// Position of the character in seconds.
        /// </summary>
        internal unsafe float start_time;
    }
}
