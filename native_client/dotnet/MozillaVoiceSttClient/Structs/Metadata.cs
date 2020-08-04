using System;
using System.Runtime.InteropServices;

namespace MozillaVoiceSttClient.Structs
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct Metadata
    {
        /// <summary>
        /// Native list of candidate transcripts.
        /// </summary>
        internal unsafe IntPtr transcripts;
        /// <summary>
        /// Count of transcripts from the native side.
        /// </summary>
        internal unsafe int num_transcripts;
    }
}
