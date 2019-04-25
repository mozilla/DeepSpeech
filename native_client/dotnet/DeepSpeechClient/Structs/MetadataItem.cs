using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient.Structs
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct MetadataItem
    {
        /// <summary>
        /// Native character.
        /// </summary>
        internal unsafe IntPtr character;
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
