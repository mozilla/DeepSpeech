using System.Runtime.InteropServices;
using Alphabet = System.IntPtr;
using Scorer = System.IntPtr;
using Session = System.IntPtr;
using MemmappedEnv = System.IntPtr;
using GraphDef = System.IntPtr;

namespace DeepSpeechClient.Structs
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public unsafe struct ModelState
    {
        public MemmappedEnv mmap_env;
        public Session session;
        public GraphDef graph_def;
        public uint ncep;
        public uint ncontext;
        public Alphabet alphabet;
        public Scorer scorer;
        public uint beam_width;
        public uint n_steps;
        public uint mfcc_feats_per_timestep;
        public uint n_context;
    }
}
