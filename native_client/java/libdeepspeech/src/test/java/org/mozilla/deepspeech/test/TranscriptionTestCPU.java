package org.mozilla.deepspeech.test;

import org.junit.Test;
import org.mozilla.deepspeech.libraryloader.DeepSpeechLibraryConfig;
import org.mozilla.deepspeech.recognition.DeepSpeechModel;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.EOFException;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.mozilla.deepspeech.test.WorkingDirectory.WORKING_DIRECTORY;

public class TranscriptionTestCPU {

    private static final int N_CEP = 26;
    private static final int N_CONTEXT = 9;
    private static final int BEAM_WIDTH = 50;
    private static final float LM_ALPHA = 0.75f;
    private static final float LM_BETA = 1.85f;

    @Test
    public void testTranscriptionCPU() throws Exception {
        System.out.println("When running this test, make sure the working directory of the test application is setup like this:\n\nworkingDirectory\n\tmodel\n\t\toutput_graph.pb\n\t\talphabet.txt\n\t\tlm.binary\n\tnatives\n\t\tlibdeepspeech.so\n\t\tlibdeepspeech-jni.so\n\ttestInput.wav\n");
        DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Desktop.CPU(
                new File(WORKING_DIRECTORY, "natives/libdeepspeech_cpu.so").toURI().toURL(),
                new File(WORKING_DIRECTORY, "natives/libdeepspeech-jni_cpu.so").toURI().toURL()
        );
        config.loadDeepSpeech();

        AudioInputStream audioIn = AudioSystem.getAudioInputStream(new File(WORKING_DIRECTORY, "testInput.wav"));
        DeepSpeechModel model = new DeepSpeechModel(new File(WORKING_DIRECTORY, "model/output_graph.pb"), N_CEP, N_CONTEXT, new File(WORKING_DIRECTORY, "model/alphabet.txt"), BEAM_WIDTH);
        model.enableLMLanguageModel(new File(WORKING_DIRECTORY, "model/alphabet.txt"), new File(WORKING_DIRECTORY, "model/lm.binary"), new File(WORKING_DIRECTORY, "model/trie"), LM_ALPHA, LM_BETA);

        long sampleRate = (long) audioIn.getFormat().getSampleRate();
        long numSamples = audioIn.getFrameLength();
        byte[] bytes = new byte[(int) (numSamples * 2)];
        if (audioIn.read(bytes) == -1)
            throw new EOFException();
        assertSame("Unexpected encoding", audioIn.getFormat().getEncoding(), AudioFormat.Encoding.PCM_SIGNED);
        ByteBuffer audioBuffer = ByteBuffer.allocateDirect((int) (numSamples * 2)).order(ByteOrder.LITTLE_ENDIAN); // 2 bytes for each sample --> 16 bit audio
        audioBuffer.put(bytes);
        String transcription = model.doSpeechToText(audioBuffer, numSamples, sampleRate);
        assertEquals("she had your dark suit in greasy wash water all year", transcription);
    }
}
