using DeepSpeechClient.Structs;
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace DeepSpeechClient.Extensions
{
    internal static class NativeExtensions
    {
        /// <summary>
        /// Converts native pointer to UTF-8 encoded string.
        /// </summary>
        /// <param name="intPtr">Native pointer.</param>
        /// <param name="releasePtr">Optional parameter to release the native pointer.</param>
        /// <returns>Result string.</returns>
        internal static string PtrToString(this IntPtr intPtr, bool releasePtr = true)
        {
            int len = 0;
            while (Marshal.ReadByte(intPtr, len) != 0) ++len;
            byte[] buffer = new byte[len];
            Marshal.Copy(intPtr, buffer, 0, buffer.Length);
            if (releasePtr)
                NativeImp.STT_FreeString(intPtr);
            string result = Encoding.UTF8.GetString(buffer);
            return result;
        }

        /// <summary>
        /// Converts a pointer into managed TokenMetadata object.
        /// </summary>
        /// <param name="intPtr">Native pointer.</param>
        /// <returns>TokenMetadata managed object.</returns>
        private static Models.TokenMetadata PtrToTokenMetadata(this IntPtr intPtr)
        {
            var token = Marshal.PtrToStructure<TokenMetadata>(intPtr);
            var managedToken = new Models.TokenMetadata
            {
                Timestep = token.timestep,
                StartTime = token.start_time,
                Text = token.text.PtrToString(releasePtr: false)
            };
            return managedToken;
        }

        /// <summary>
        /// Converts a pointer into managed CandidateTranscript object.
        /// </summary>
        /// <param name="intPtr">Native pointer.</param>
        /// <returns>CandidateTranscript managed object.</returns>
        private static Models.CandidateTranscript PtrToCandidateTranscript(this IntPtr intPtr)
        {
            var managedTranscript = new Models.CandidateTranscript();
            var transcript = Marshal.PtrToStructure<CandidateTranscript>(intPtr);

            managedTranscript.Tokens = new Models.TokenMetadata[transcript.num_tokens];
            managedTranscript.Confidence = transcript.confidence;

            //we need to manually read each item from the native ptr using its size
            var sizeOfTokenMetadata = Marshal.SizeOf(typeof(TokenMetadata));
            for (int i = 0; i < transcript.num_tokens; i++)
            {
                managedTranscript.Tokens[i] = transcript.tokens.PtrToTokenMetadata();
                transcript.tokens += sizeOfTokenMetadata;
            }

            return managedTranscript;
        }

        /// <summary>
        /// Converts a pointer into managed Metadata object.
        /// </summary>
        /// <param name="intPtr">Native pointer.</param>
        /// <returns>Metadata managed object.</returns>
        internal static Models.Metadata PtrToMetadata(this IntPtr intPtr)
        {
            var managedMetadata = new Models.Metadata();
            var metadata = Marshal.PtrToStructure<Metadata>(intPtr);

            managedMetadata.Transcripts = new Models.CandidateTranscript[metadata.num_transcripts];

            //we need to manually read each item from the native ptr using its size
            var sizeOfCandidateTranscript = Marshal.SizeOf(typeof(CandidateTranscript));
            for (int i = 0; i < metadata.num_transcripts; i++)
            {
                managedMetadata.Transcripts[i] = metadata.transcripts.PtrToCandidateTranscript();
                metadata.transcripts += sizeOfCandidateTranscript;
            }

            NativeImp.STT_FreeMetadata(intPtr);
            return managedMetadata;
        }
    }
}
