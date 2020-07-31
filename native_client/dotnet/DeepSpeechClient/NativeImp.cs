using DeepSpeechClient.Enums;

using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient
{
    /// <summary>
    /// Wrapper for the native implementation of "libaeiou.so"
    /// </summary>
    internal static class NativeImp
    {
        #region Native Implementation
        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static extern IntPtr STT_Version();

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern ErrorCodes STT_CreateModel(string aModelPath,
            ref IntPtr** pint);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern IntPtr STT_ErrorCodeToErrorMessage(int aErrorCode);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern uint STT_GetModelBeamWidth(IntPtr** aCtx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern ErrorCodes STT_SetModelBeamWidth(IntPtr** aCtx,
            uint aBeamWidth);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern ErrorCodes STT_CreateModel(string aModelPath,
            uint aBeamWidth,
            ref IntPtr** pint);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal unsafe static extern int STT_GetModelSampleRate(IntPtr** aCtx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ErrorCodes STT_EnableExternalScorer(IntPtr** aCtx,
            string aScorerPath);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ErrorCodes STT_DisableExternalScorer(IntPtr** aCtx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ErrorCodes STT_SetScorerAlphaBeta(IntPtr** aCtx,
            float aAlpha,
            float aBeta);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static unsafe extern IntPtr STT_SpeechToText(IntPtr** aCtx,
            short[] aBuffer,
            uint aBufferSize);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl, SetLastError = true)]
        internal static unsafe extern IntPtr STT_SpeechToTextWithMetadata(IntPtr** aCtx,
            short[] aBuffer,
            uint aBufferSize,
            uint aNumResults);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void STT_FreeModel(IntPtr** aCtx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ErrorCodes STT_CreateStream(IntPtr** aCtx,
               ref IntPtr** retval);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void STT_FreeStream(IntPtr** aSctx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void STT_FreeMetadata(IntPtr metadata);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void STT_FreeString(IntPtr str);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static unsafe extern void STT_FeedAudioContent(IntPtr** aSctx,
            short[] aBuffer,
            uint aBufferSize);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern IntPtr STT_IntermediateDecode(IntPtr** aSctx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern IntPtr STT_IntermediateDecodeWithMetadata(IntPtr** aSctx,
            uint aNumResults);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi, SetLastError = true)]
        internal static unsafe extern IntPtr STT_FinishStream(IntPtr** aSctx);

        [DllImport("libaeiou.so", CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern IntPtr STT_FinishStreamWithMetadata(IntPtr** aSctx,
            uint aNumResults);
        #endregion
    }
}
