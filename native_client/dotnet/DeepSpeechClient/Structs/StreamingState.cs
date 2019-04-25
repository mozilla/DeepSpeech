using System.Runtime.InteropServices;

namespace DeepSpeechClient.Structs
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal unsafe struct StreamingState
    {
        public float last_sample; // used for preemphasis
        public bool skip_next_mfcc;
        public ModelState* model;
    }
}
