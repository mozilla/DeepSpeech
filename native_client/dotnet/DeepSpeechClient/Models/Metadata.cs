namespace DeepSpeechClient.Models
{
    /// <summary>
    /// Stores the entire CTC output as an array of character metadata objects.
    /// </summary>
    public class Metadata
    {
        /// <summary>
        /// Approximated probability (confidence value) for this transcription.
        /// </summary>
        public double Probability { get; set; }
        /// <summary>
        /// List of metada items containing char, timespet, and time offset.
        /// </summary>
        public MetadataItem[] Items { get; set; }
    }
}