namespace MozillaVoiceSttClient.Models
{
    /// <summary>
    /// Stores the entire CTC output as an array of character metadata objects.
    /// </summary>
    public class CandidateTranscript
    {
        /// <summary>
        /// Approximated confidence value for this transcription.
        /// </summary>
        public double Confidence { get; set; }
        /// <summary>
        /// List of metada tokens containing text, timestep, and time offset.
        /// </summary>
        public TokenMetadata[] Tokens { get; set; }
    }
}