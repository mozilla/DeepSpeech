namespace DeepSpeechClient.Models
{
    /// <summary>
    /// Stores the entire CTC output as an array of character metadata objects.
    /// </summary>
    public class Metadata
    {
        /// <summary>
        /// List of candidate transcripts.
        /// </summary>
        public CandidateTranscript[] Transcripts { get; set; }
    }
}