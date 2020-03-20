namespace DeepSpeechClient.Models
{
    /// <summary>
    /// Stores each individual character, along with its timing information.
    /// </summary>
    public class TokenMetadata
    {
        /// <summary>
        /// Char of the current timestep.
        /// </summary>
        public string Text;
        /// <summary>
        /// Position of the character in units of 20ms.
        /// </summary>
        public int Timestep;
        /// <summary>
        /// Position of the character in seconds.
        /// </summary>
        public float StartTime;
    }
}