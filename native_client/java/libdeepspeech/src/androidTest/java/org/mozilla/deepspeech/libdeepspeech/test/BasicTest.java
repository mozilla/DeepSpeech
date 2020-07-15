package org.mozilla.deepspeech.libdeepspeech.test;

import android.content.Context;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.FixMethodOrder;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;

import static org.junit.Assert.*;

import org.mozilla.deepspeech.libdeepspeech.DeepSpeechModel;
import org.mozilla.deepspeech.libdeepspeech.CandidateTranscript;

import java.io.RandomAccessFile;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class BasicTest {

    public static final String modelFile    = "/data/local/tmp/test/output_graph.tflite";
    public static final String scorerFile   = "/data/local/tmp/test/kenlm.scorer";
    public static final String wavFile      = "/data/local/tmp/test/LDC93S1.wav";

    private char readLEChar(RandomAccessFile f) throws IOException {
        byte b1 = f.readByte();
        byte b2 = f.readByte();
        return (char)((b2 << 8) | b1);
    }

    private int readLEInt(RandomAccessFile f) throws IOException {
        byte b1 = f.readByte();
        byte b2 = f.readByte();
        byte b3 = f.readByte();
        byte b4 = f.readByte();
        return (int)((b1 & 0xFF) | (b2 & 0xFF) << 8 | (b3 & 0xFF) << 16 | (b4 & 0xFF) << 24);
    }

    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getTargetContext();

        assertEquals("org.mozilla.deepspeech.libdeepspeech.test", appContext.getPackageName());
    }

    @Test
    public void loadDeepSpeech_basic() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile);
        m.freeModel();
    }

    private String candidateTranscriptToString(CandidateTranscript t) {
        String retval = "";
        for (int i = 0; i < t.getNumTokens(); ++i) {
            retval += t.getToken(i).getText();
        }
        return retval;
    }

    private String doSTT(DeepSpeechModel m, boolean extendedMetadata) {
        try {
            RandomAccessFile wave = new RandomAccessFile(wavFile, "r");

            wave.seek(20); char audioFormat = this.readLEChar(wave);
            assert (audioFormat == 1); // 1 is PCM

            wave.seek(22); char numChannels = this.readLEChar(wave);
            assert (numChannels == 1); // MONO

            wave.seek(24); int sampleRate = this.readLEInt(wave);
            assert (sampleRate == 16000); // 16000 Hz

            wave.seek(34); char bitsPerSample = this.readLEChar(wave);
            assert (bitsPerSample == 16); // 16 bits per sample

            wave.seek(40); int bufferSize = this.readLEInt(wave);
            assert (bufferSize > 0);

            wave.seek(44);
            byte[] bytes = new byte[bufferSize];
            wave.readFully(bytes);

            short[] shorts = new short[bytes.length/2];
            // to turn bytes to shorts as either big endian or little endian.
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);

            if (extendedMetadata) {
                return candidateTranscriptToString(m.sttWithMetadata(shorts, shorts.length, 1).getTranscript(0));
            } else {
                return m.stt(shorts, shorts.length);
            }
        } catch (FileNotFoundException ex) {

        } catch (IOException ex) {

        } finally {

        }

        return "";
    }

    @Test
    public void loadDeepSpeech_stt_noLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile);

        String decoded = doSTT(m, false);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.freeModel();
    }

    @Test
    public void loadDeepSpeech_stt_withLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile);
        m.enableExternalScorer(scorerFile);

        String decoded = doSTT(m, false);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.freeModel();
    }

    @Test
    public void loadDeepSpeech_sttWithMetadata_noLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile);

        String decoded = doSTT(m, true);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.freeModel();
    }

    @Test
    public void loadDeepSpeech_sttWithMetadata_withLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile);
        m.enableExternalScorer(scorerFile);

        String decoded = doSTT(m, true);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.freeModel();
    }
}
