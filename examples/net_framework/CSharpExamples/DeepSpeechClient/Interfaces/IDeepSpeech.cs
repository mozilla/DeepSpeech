namespace DeepSpeechClient.Interfaces
{
    public interface IDeepSpeech
    {
        void PrintVersions();

        unsafe int CreateModel(string aModelPath, uint aNCep,
                   uint aNContext,
                   string aAlphabetConfigPath,
                   uint aBeamWidth);

        unsafe int EnableDecoderWithLM(string aAlphabetConfigPath,
                  string aLMPath,
                  string aTriePath,
                  float aLMAlpha,
                  float aLMBeta);

        unsafe string SpeechToText(short[] aBuffer,
                uint aBufferSize,
                uint aSampleRate);

        unsafe void DiscardStream();

        unsafe int SetupStream(uint aPreAllocFrames, uint aSampleRate);

        unsafe void FeedAudioContent(short[] aBuffer, uint aBufferSize);

        unsafe string IntermediateDecode();

        unsafe string FinishStream();
    }
}
