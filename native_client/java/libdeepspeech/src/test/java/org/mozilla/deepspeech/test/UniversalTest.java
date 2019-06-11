package org.mozilla.deepspeech.test;

import org.junit.Test;
import org.mozilla.deepspeech.libraryloader.DeepSpeechLibraryConfig;
import org.mozilla.deepspeech.recognition.DeepSpeechModel;
import org.mozilla.deepspeech.recognition.stream.SpeechRecognitionAudioStream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.EOFException;
import java.io.File;

import static org.mozilla.deepspeech.test.WorkingDirectory.WORKING_DIRECTORY;

public class UniversalTest {

    private static final int N_CEP = 26;
    private static final int N_CONTEXT = 9;
    private static final int BEAM_WIDTH = 50;
    private static final float LM_ALPHA = 0.75f;
    private static final float LM_BETA = 1.85f;

    @Test
    public void testDeepSpeech() throws Exception {
        System.out.println("When running this test, make sure the working directory of the test application is setup like this:\n\nworkingDirectory\n\tmodel\n\t\toutput_graph.pb\n\t\talphabet.txt\n\t\tlm.binary\n\tnatives\n\t\tlibdeepspeech.so\n\t\tlibdeepspeech-jni.so\n\ttestInput.wav\n");

        DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Desktop.Cuda(
                new File(WORKING_DIRECTORY, "natives/libdeepspeech_cuda.so").toURI().toURL(),
                new File(WORKING_DIRECTORY, "natives/libdeepspeech-jni_cuda.so").toURI().toURL(),
                new File(System.getenv("CUDA_PATH"), "lib64/libcudart.so").toURI().toURL() // Make sure CUDA_PATH is defined
        );
//        DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Desktop.CPU(
//                new File("natives/libdeepspeech_cpu.so").toURI().toURL(),
//                new File("natives/libdeepspeech-jni_cpu.so").toURI().toURL()
//        );
        config.loadDeepSpeech();

        AudioInputStream audioIn = AudioSystem.getAudioInputStream(new File(WORKING_DIRECTORY, "testInput.wav"));
        DeepSpeechModel model = new DeepSpeechModel(new File(WORKING_DIRECTORY, "model/output_graph.pb"), N_CEP, N_CONTEXT, new File(WORKING_DIRECTORY, "model/alphabet.txt"), BEAM_WIDTH);
        model.enableLMLanguageModel(new File(WORKING_DIRECTORY, "model/alphabet.txt"), new File(WORKING_DIRECTORY, "model/lm.binary"), new File(WORKING_DIRECTORY, "model/trie"), LM_ALPHA, LM_BETA);

        long sampleRate = (long) audioIn.getFormat().getSampleRate();
        long numSamples = audioIn.getFrameLength();
        long frameSize = audioIn.getFormat().getFrameSize();

        assert audioIn.getFormat().getSampleSizeInBits() == 16; // 16 bit samples
        assert audioIn.getFormat().getEncoding() == AudioFormat.Encoding.PCM_SIGNED;
        assert sampleRate == 16000;

//        ByteBuffer audioBuffer = ByteBuffer.allocateDirect((int) (numSamples * 2)); // 2 bytes for each sample --> 16 bit audio
//        String transcription = model.doSpeechToText(audioBuffer, numSamples, sampleRate);
//

        SpeechRecognitionAudioStream stream = new SpeechRecognitionAudioStream(sampleRate);
        {
            int numBytes = (int) (numSamples * frameSize);
            byte[] bytes = new byte[numBytes];
            if (audioIn.read(bytes) == -1)
                throw new EOFException(); // End of stream reached

            stream.write(bytes, 0, bytes.length / 2);
            System.out.println(stream.doSpeechToText(model));
            stream.write(bytes, bytes.length / 2, bytes.length / 2);
            System.out.println(stream.doSpeechToText(model));
            stream.clear();
            stream.write(bytes, 0, bytes.length / 2);
            System.out.println(stream.doSpeechToText(model));
            stream.write(bytes, bytes.length / 2, bytes.length / 2);
            System.out.println(stream.doSpeechToText(model));
        }

    }
}