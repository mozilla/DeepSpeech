#DeepSpeech Java Binding##

This are the Java bindings to use the DeepSpeech library from Java.

##How to use##

Your Java application needs to load two native libraries: `libdeepspeech` and `libdeepspeech-jni`.

__If you do not load these libraries, the invocation of the native functions will throw an `UnsatisfiedLinkError`.__

Both are shared library which can be found in the DeepSpeech native client releases. 
Make sure that binaries for all OSs and configurations _(OS, CPU / Cuda)_ your application will run on are included in your application. _(eg. classpath, installation directory)_.

> See [DeepSpeech releases]

Loading the native necessary library from your Java Application is done using `DeepSpeechLibraryConfig`.

There are two main types of library configurations:

* Desktop
    * CPU
    * CUDA
* Android

On a desktop pc (except OSX for now) cuda acceleration is supported. This requires a compatible GPU and Driver.

###The Desktop CPU configuration###

```java
DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Desktop.CPU(
        new File("natives/" + getOS() + "/libdeepspeech_cpu." + getOSLibExtension()).toURI().toURL(),
        new File("natives/" + getOSName() + "/libdeepspeech-jni_cpu." + getOSLibExtension()).toURI().toURL()
);
try {
     config.loadDeepSpeech();
} catch (IOException e) {
    Logger.logMsg(Logger.ERROR, "Failed to load DeepSpeech!");
}
```
Make sure that `libdeepspeech-java.jar`, which can be found in the [DeepSpeech releases], is appended to your classpath eg. by compiling the Java library using gradle or maven:

```gradle
implementation 'org.mozilla.deepspeech:deepspeech-java:1.0'
```

This library loading configuration is used to load the DeepSpeech native libraries for desktop devices without cuda support.

It is recommended to try loading the Cuda library first, and fall back on the CPU library, if this fails due to an incompatible device.

The two URLs of the constructor must point to the library resource as shown here.
Make sure not to hardcode the OS with this configuration.

Loading the library may throw a `WrongJNILibraryException`. This means that that a different jni configuration is expected.
A jni library compiled for Cuda usage could eg. be mismatched with a jni library configuration designed to load a CPU jni library.


###The Desktop Cuda configuration###

```java
DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Desktop.Cuda(
    new File("natives/" + getOSName() + "/libdeepspeech_cuda." + getOSLibExtension()).toURI().toURL(),
    new File("natives/" + getOSName() + "/libdeepspeech-jni_cuda." + getOSLibextension()).toURI().toURL(),
    new File(System.getenv("CUDA_PATH"), "lib64/libcudart.so").toURI().toURL() // CUDA_PATH is normally defined on installation of the Cuda computing toolkit
);
try {
    config.loadDeepSpeech();
} catch (CudaNoDeviceError e) {
    JOptionPane.showMessageDialog(frame, "You do not have a CUDA supporting CPU!");
    loadCPUConfigurationInstead(); // Fall back on cpu library    
} catch (CudaInsufficientDriverError e) {
    JOptionPane.showMessageDialog(frame, "Update your driver!");
    loadCPUConfigurationInstead(); // Fall back on cpu library
}
```

The third file of the configuration is the URL pointing to the Cuda runtime library
of the driver. Please write your own better logic find this library.

__Keep in mind, that the user might not have a CUDA supporting GPU or an up-to-date driver. Make sure to fall back on the
cpu library in case the cuda runtime could not be created!__

When using the Cuda configuration, you have access to a method called `Cuda.getDevice()`.
It returns a `Device` object storing information about the GPU used for the runtime that might be useful: 

####The string representation of an example GPU####

```
Device {
	name='GeForce GTX 980 Ti',
	uuid='[-68, 53, 107, -15, 63, -59, -64, 73, 100, 5, 117, 79, -121, -2, 37, 109]',
	totalGlobalMemory=5.9 GiB,
	sharedMemoryPerBlock=48.0 KiB,
	registerPerBlock=65536,
	warpSize=32,
	memoryPitch=2.0 GiB,
	maxThreadsPerBlock=1024,
	maxThreadsDim=[1024, 1024, 64],
	maxGridSize=[2147483647, 65535, 65535],
	totalConstantMemory=64.0 KiB,
	major=5,
	minor=2,
	clockRate=1228000KHz,
	textureAlignment=512,
	deviceOverlap=true,
	multiProcessorCount=22,
	kernelExecTimeoutEnabled=true,
	integrated=false,
	canMapHostMemory=true,
	computeMode=DEFAULT
}
```

###Android configuration###

```java
DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Android();
try {
    config.loadDeepSpeech();
} catch (IOException e) {
    // Never thrown on Android
    Logger.logMsg(Logger.ERROR, "Failed to load DeepSpeech!");
} catch (UnsatisfiedLinkError e){
    Logger.logMsg(Logger.ERROR, "The user uses a device with an unsupported ABI!");    
}
```

DeepSpeech supports the following Android [ABIs]: `arm64-v8a`, `armeabi-v7a`, `x86_64`.
If the user's device has an unsupported ABI, an `UnsatisfiedLinkError` will be thrown.

Make sure to implement the `deepspeechandroid.aar` in the app gradle script, which can also be found in the [DeepSpeech releases].

```gradle
implementation 'org.mozilla.deepspeech:deepspeech-android:1.0' 
```

If you compile the normal Java library, place the native `libdeepspeech` and `libdeepspeech-jni` libraries for all supported [ABIs] under `src/main/jniLibs`.
For more information, read the Readme of `DeepSpeech-jni`.

###Tutorial###
After the native libraries are loaded, the bindings are ready to be used in your Java application.


####Loading a model####

Loading a model is done the following:

```java
DeepSpeechModel model = new DeepSpeechModel(new File("model/output_graph.pb"), N_CEP, N_CONTEXT, new File("model/alphabet.txt"), BEAM_WIDTH);
```

You can also enable the LM language model for your model:
```java
model.enableLMLanguageModel(new File("model/alphabet.txt"), new File("model/lm.binary"), new File("model/trie"), LM_ALPHA, LM_BETA);
```
####Audio transcription####
Transcribing audio is done the following
#####Example#####
```java

AudioInputStream audioIn = AudioSystem.getAudioInputStream(new File("testInput.wav"));

long sampleRate = (long) audioIn.getFormat().getSampleRate();
long numSamples = audioIn.getFrameLength();
long frameSize = audioIn.getFormat().getFrameSize();        

ByteBuffer audioBuffer = ByteBuffer.allocateDirect((int) (numSamples * 2)); // 2 bytes for each sample --> 16 bit audio
String transcription = model.doSpeechToText(audioBuffer, numSamples, sampleRate);

```

The method `doSpeechToText` needs the following parameters:
 * A directly allocated byte buffer audio buffer that contains `numSamples` amount of 16 bit (2 byte) samples. (Little endian)
 * The amount of samples contained in the buffer
 * the sample rate of the audio signal
#####Speech Recognition Result Metadata#####

To get additional information about the speech-recognition result than just the transcription string, use the method `doSpeechRecognitionWithMeta`.
This method returns a `SpeechRecognitionResult` instead of a `String`.

######Properties of `SpeechRecognitionResult`######

* double confidence
* Collection<SpokenCharacterData> spokenCharacters
* String text

######Properties of `SpokenCharacterData`######

* char character
* long startTime

####Speech Recognition Audio Stream####
 
 When the audio has continuous input, consider using a `SpeechRecognitionAudioStream` where audio content can be continuously written to.
 Speech recognition can be invoked any time to transcribe the audio content written to the stream. When the stream exceeds some size threshold,
 the old audio content can be cleared. After clearing, the audio buffer is empty again.
 
#####Example#####
 ```java
SpeechRecognitionAudioStream stream = new SpeechRecognitionAudioStream(sampleRate);

int numBytes = (int) (numSamples * frameSize);
byte[] bytes = new byte[numBytes]; // The audio buffer

audioIn.read(bytes); // Populating buffer with audio data

stream.write(bytes, 0, bytes.length / 2);
System.out.println(stream.doSpeechToText(model));

stream.write(bytes, bytes.length / 2, bytes.length / 2);
System.out.println(stream.doSpeechToText(model));

stream.clear();

stream.write(bytes, 0, bytes.length / 2);
System.out.println(stream.doSpeechToText(model));

stream.write(bytes, bytes.length / 2, bytes.length / 2);
System.out.println(stream.doSpeechToText(model));
```
#####Output#####
```
she had you
she had your dark suit in greasy wash water all year
she had you
she had your dark suit in greasy wash water all year
```

##Running the tests##


###Setting up the working directory###

The tests must be run in a special working directory in order to pass.
This working directory must be placed in the project root directory with the name `workingDirectory`, 
or change the constant `WORKING_DIRECTORY` in the class `WorkingDirectory` located in the java tests under package `org.mozilla.deepspeech.test` to your
desired location (Not recommended).

####Structure of the working directory####

> See [Smoke Test]
* model **The LDC93S1 model**
    * output_graph.pb
	* alphabet.txt
	* lm.binary
	* trie
* natives **Natives for your platform**
	* libdeepspeech
	* libdeepspeech-jni
* testInput.wav **Download: [LDC93S1.wav]**

##Building the `libdeepspeech-java.jar`##

To build the library, you need to have gradle installed.
Instructions on how to install gradle: [Gradle installation]

Run the following command in the project root directory `libdeepspeech`:

```
gradle build -x test
```

We do not want to run the unit tests as they will fail, because each one needs to be run in a different JVM as
native libraries cannot be loaded twice or unloaded. Your built `libdeepspeech.jar` file will be placed in `build/libs`.

[ABIs]: https://developer.android.com/ndk/guides/abis
[Smoke Test]: https://github.com/mozilla/DeepSpeech/tree/master/data/smoke_test
[LDC93S1.wav]: https://github.com/mozilla/DeepSpeech/blob/master/data/smoke_test/LDC93S1.wav
[Gradle installation]: https://gradle.org/install/
[DeepSpeech releases]: https://github.com/mozilla/DeepSpeech/releases
