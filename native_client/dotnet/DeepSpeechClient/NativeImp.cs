using DeepSpeechClient.Enums;
using DeepSpeechClient.Structs;

using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient
{
    /// <summary>
    /// Wrapper for the native implementation of "libdeepspeech.so"
    /// </summary>
    internal static class NativeImp
    {
        #region Native Implementation
        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DS_PrintVersions();

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern ErrorCodes DS_CreateModel(string aModelPath,
                   uint aNCep,
                   uint aNContext,
                   string aAlphabetConfigPath,
                   uint aBeamWidth,
                   ref ModelState** pint);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ErrorCodes DS_EnableDecoderWithLM(ModelState** aCtx,
                  string aAlphabetConfigPath,
                  string aLMPath,
                  string aTriePath,
                  float aLMAlpha,
                  float aLMBeta);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static unsafe extern IntPtr DS_SpeechToText(ModelState** aCtx,
                 short[] aBuffer,
                uint aBufferSize,
                uint aSampleRate);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl, SetLastError = true)]
        internal static unsafe extern IntPtr DS_SpeechToTextWithMetadata(ModelState** aCtx,
                 short[] aBuffer,
                uint aBufferSize,
                uint aSampleRate);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void DS_DestroyModel(ModelState** aCtx);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ErrorCodes DS_SetupStream(ModelState** aCtx,
               uint aPreAllocFrames,
               uint aSampleRate, ref StreamingState** retval);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void DS_DiscardStream(ref StreamingState** aSctx);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void DS_FreeMetadata(IntPtr metadata);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void DS_FreeString(IntPtr str);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static unsafe extern void DS_FeedAudioContent(StreamingState** aSctx,
                     short[] aBuffer,
                    uint aBufferSize);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern string DS_IntermediateDecode(StreamingState** aSctx);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static unsafe extern IntPtr DS_FinishStream(  StreamingState** aSctx);

        [DllImport("libdeepspeech.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern IntPtr DS_FinishStreamWithMetadata(StreamingState** aSctx);
        #endregion
    }
}
