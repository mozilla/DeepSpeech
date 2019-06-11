package org.mozilla.deepspeech;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import org.mozilla.deepspeech.libraryloader.DeepSpeechLibraryConfig;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Android();
        try {
            config.loadDeepSpeech();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        DeepSpeech.printVersions();
        setContentView(R.layout.activity_main);
    }


}
