package org.mozilla.deepspeech.recognition.stream;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.recognition.DeepSpeechModel;
import org.mozilla.deepspeech.recognition.SpeechRecognitionResult;

/**
 * Represents a 16 bit audio output stream where eg. the input of a microphone input stream are written onto. Intermediate speech recognition calls can be performed
 */
public class SpeechRecognitionAudioStream extends NativeByteArrayOutputStream {

    /**
     * The sample-rate of the audio signal
     */
    private final long sampleRate;

    /**
     * @param sampleRate the sample-rate of the audio signal
     */
    public SpeechRecognitionAudioStream(long sampleRate) {
        this.sampleRate = sampleRate;
    }

    @NotNull
    public String doSpeechToText(@NotNull DeepSpeechModel model) {
        return model.doSpeechToTextUnsafe(this.address(), this.getStreamSize() / 2, this.sampleRate);
    }

    @NotNull
    public SpeechRecognitionResult doSpeechRecognitionWithMeta(@NotNull DeepSpeechModel model) {
        return model.doSpeechRecognitionWithMetaUnsafe(this.address(), this.getStreamSize() / 2, this.sampleRate);
    }

    public long getSampleRate() {
        return sampleRate;
    }

    /**
     * Clears the audio buffer where the audio data is written to.
     */
    public void flushOldAudio() {
        clear();
    }
}
