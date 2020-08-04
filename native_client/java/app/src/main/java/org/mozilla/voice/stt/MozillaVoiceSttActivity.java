package org.mozilla.voice.stt;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import android.view.View;
import android.widget.TextView;
import android.widget.EditText;
import android.widget.Button;

import android.media.MediaPlayer;

import java.io.RandomAccessFile;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;

import org.mozilla.voice.stt.MozillaVoiceSttModel;

public class MozillaVoiceSttActivity extends AppCompatActivity {

    MozillaVoiceSttModel _m = null;

    EditText _tfliteModel;
    EditText _audioFile;

    TextView _decodedString;
    TextView _tfliteStatus;

    Button _startInference;

    final int BEAM_WIDTH = 50;

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

    private void newModel(String tfliteModel) {
        this._tfliteStatus.setText("Creating model");
        if (this._m == null) {
            // sphinx-doc: java_ref_model_start
            this._m = new MozillaVoiceSttModel(tfliteModel);
            this._m.setBeamWidth(BEAM_WIDTH);
            // sphinx-doc: java_ref_model_stop
        }
    }

    private void doInference(String audioFile) {
        long inferenceExecTime = 0;

        this._startInference.setEnabled(false);

        this.newModel(this._tfliteModel.getText().toString());

        this._tfliteStatus.setText("Extracting audio features ...");

        try {
            RandomAccessFile wave = new RandomAccessFile(audioFile, "r");

            wave.seek(20); char audioFormat = this.readLEChar(wave);
            assert (audioFormat == 1); // 1 is PCM
            // tv_audioFormat.setText("audioFormat=" + (audioFormat == 1 ? "PCM" : "!PCM"));

            wave.seek(22); char numChannels = this.readLEChar(wave);
            assert (numChannels == 1); // MONO
            // tv_numChannels.setText("numChannels=" + (numChannels == 1 ? "MONO" : "!MONO"));

            wave.seek(24); int sampleRate = this.readLEInt(wave);
            assert (sampleRate == this._m.sampleRate()); // desired sample rate
            // tv_sampleRate.setText("sampleRate=" + (sampleRate == 16000 ? "16kHz" : "!16kHz"));

            wave.seek(34); char bitsPerSample = this.readLEChar(wave);
            assert (bitsPerSample == 16); // 16 bits per sample
            // tv_bitsPerSample.setText("bitsPerSample=" + (bitsPerSample == 16 ? "16-bits" : "!16-bits" ));

            wave.seek(40); int bufferSize = this.readLEInt(wave);
            assert (bufferSize > 0);
            // tv_bufferSize.setText("bufferSize=" + bufferSize);

            wave.seek(44);
            byte[] bytes = new byte[bufferSize];
            wave.readFully(bytes);

            short[] shorts = new short[bytes.length/2];
            // to turn bytes to shorts as either big endian or little endian.
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);

            this._tfliteStatus.setText("Running inference ...");

            long inferenceStartTime = System.currentTimeMillis();

            // sphinx-doc: java_ref_inference_start
            String decoded = this._m.stt(shorts, shorts.length);
            // sphinx-doc: java_ref_inference_stop

            inferenceExecTime = System.currentTimeMillis() - inferenceStartTime;

            this._decodedString.setText(decoded);

        } catch (FileNotFoundException ex) {

        } catch (IOException ex) {

        } finally {

        }

        this._tfliteStatus.setText("Finished! Took " + inferenceExecTime + "ms");

        this._startInference.setEnabled(true);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_deep_speech);

        this._decodedString = (TextView) findViewById(R.id.decodedString);
        this._tfliteStatus = (TextView) findViewById(R.id.tfliteStatus);

        this._tfliteModel   = (EditText) findViewById(R.id.tfliteModel);
        this._audioFile     = (EditText) findViewById(R.id.audioFile);

        this._tfliteModel.setText("/sdcard/deepspeech/output_graph.tflite");
        this._tfliteStatus.setText("Ready, waiting ...");

        this._audioFile.setText("/sdcard/deepspeech/audio.wav");

        this._startInference = (Button) findViewById(R.id.btnStartInference);
    }

    public void onClick_inference_handler(View v) {
        this.playAudioFile();
        this.doInference(this._audioFile.getText().toString());
    }

    public void playAudioFile() {
        try {
            MediaPlayer mediaPlayer = new  MediaPlayer();
            mediaPlayer.setDataSource(this._audioFile.getText().toString());
            mediaPlayer.prepare();
            mediaPlayer.start();
        } catch (IOException ex) {

        }
    }

    public void onClick_audio_handler(View v) {
        this.playAudioFile();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (this._m != null) {
            this._m.freeModel();
        }
    }
}
