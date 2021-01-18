# Java bindings
## Android bindings
Full project description and documentation on GitHub: [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

?

## Standalone Java Bindings for DeepSpeech
Does not rely on Android!

Full project description and documentation on GitHub: [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

### Setup
1. Get `libdeepspeech.so` by compiling or (from here)[], and copy it into `./native_client/java_standalone/libdeepspeech/libs/`
2. Run `setup.sh` as root or with `sudo`
3. You can now find the files needed for your project at `./native_client/java_standalone/build/`

### Adding DeepSpeech to your Java project
1. Copy `libdeepspeech.jar` from `./native_client/java_standalone/build/` into `{YOUR PROJECT ROOT}/libs/`
2. Copy '*.so' from `./native_client/java_standalone/build/` into `{YOUR PROJECT ROOT}/src/main/resources/jni/x86_64/`
3. Modify your `build.gradle` file to include:
```groovy
plugins {
   	id 'com.github.johnrengelman.shadow' version '5.0.0' 
}

repositories {
    flatDir {
			dirs 'libs'
    }
}

dependencies {
    compile ':libdeepspeech'
}
```
4. You can now compile your project with `./gradlew shadowJar`.

### Notes
- At this moment, this will only run on x86-64 Linux.

### Example usage
```java
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.deepspeech.libdeepspeech.DeepSpeechModel;

public class DeepSpeechExample {
    private DeepSpeechModel deepSpeechModel = null

    private final String TF_MODEL = "PATH_TO_YOUR_TENSORFLOW_MODEL";
    private final int BEAM_WIDTH = 50;

    public void createModel(String tensorFlowModel) {
        if(this.deepSpeechModel == null) {
            this.deepSpeechModel = new DeepSpeechModel(tensorFlowModel);
            this.deepSpeechModel.setBeamWidth(BEAM_WIDTH);
        }
    }

    //The format we want is a 16Khz 16 bit mono .WAV file!
    public void doInfer(String audioFile) {
        long inferenceExecTime = 0L;

        //Create a new model to use during the inference
    	this.newModel(this.TF_MODEL);
    	
    	System.out.println("Extracting audio features...");
    	
    	try {
            RandomAccessFile wave = new RandomAccessFile(audioFile, "r");

            //Assert that the audio format is PCM
            wave.seek(20);
            char audioFormat = this.readLEChar(wave);
            assert (audioFormat == 1); // 1 is PCM

            //Assert that the amount of channels is 1, meaning mono audio
            wave.seek(22);
            char numChannels = this.readLEChar(wave);
            assert (numChannels == 1); // MONO

            //Assert that the sample rate is the sample rate expected by the model
            //This can vary per model!
            wave.seek(24);
            int sampleRate = this.readLEInt(wave);
            assert (sampleRate == this.deepSpeechModel.sampleRate()); // desired sample rate

            //Assert that the bits per sample is 16
            wave.seek(34);
            char bitsPerSample = this.readLEChar(wave);
            assert (bitsPerSample == 16); // 16 bits per sample
            
            //Assert that the buffer size is more than 0
            wave.seek(40);
            int bufferSize = this.readLEInt(wave);
            assert (bufferSize > 0);

            //Read the actual contents of the audio
            wave.seek(44);
            byte[] bytes = new byte[bufferSize];
            wave.readFully(bytes);

            //Turn the byte[] into a short[] and set the correct byte order
            short[] shorts = new short[bytes.length/2];
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);
            
            System.out.println("Running inference...");

            //current time. Used later on to calculate the time the inference took
            long inferenceStartTime = System.currentTimeMillis();

            //This is where we actually do the inference
            String decoded = this.deepSpeechModel.stt(shorts, shorts.length);

            //Calculate how long it took to run the inference
            inferenceExecTime = System.currentTimeMillis() - inferenceStartTime;

            System.out.println(decoded);

        } catch (FileNotFoundException e) {
        	e.printStackTrace();
        } catch (IOException e) {
        	e.printStackTrace();
        }
    	
    	System.out.println("Finished! Took " + inferenceExecTime);
    }

    //Helper function to read a char from a RandomAccessFile in little endian
    private char readLEChar(RandomAccessFile f) throws IOException {
        byte b1 = f.readByte();
        byte b2 = f.readByte();
        return (char)((b2 << 8) | b1);
    }
    
    //Helper function to read an integer from a RandomAccessFile in little endian
    private int readLEInt(RandomAccessFile f) throws IOException {
        byte b1 = f.readByte();
        byte b2 = f.readByte();
        byte b3 = f.readByte();
        byte b4 = f.readByte();
        return (int)((b1 & 0xFF) | (b2 & 0xFF) << 8 | (b3 & 0xFF) << 16 | (b4 & 0xFF) << 24);
    }
}
```