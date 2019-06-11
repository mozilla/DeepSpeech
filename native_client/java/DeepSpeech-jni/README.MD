# DeepSpeech JNI Library

This library acts as a layer between the DeepSpeech native
library and the Java bindings. This library contains the definitions
of the native functions declared as `native` in the DeepSpeech
Java bindings.

## Building the JNI library

The JNI library can be built for desktop devices and Android
devices. If you want to build the JNI library for Android devices,
check out the Readme file of the `deepspeechandroid` project.

There are two main configurations for the JNI library

* CPU
    * > Available on desktop and Android
* Cuda
    * > Only available on desktop devices

To build a given configuration, modify the value of the `CUDA_CONFIGURATION`
variable of the `CMakeLists.txt` build file.

To build the cuda configuration go to the initial declaration of the variable
in the build file and change it to either `TRUE` or `FALSE`
depending on whether you want to build the CUDA or CPU configuration.

```cmake
set(CUDA_CONFIGURATION FALSE) # CPU configuration
```

```cmake
set(CUDA_CONFIGURATION TRUE) # CUDA configuration
```

If you are building the library for android, _(Use Android Studio and also follow instructions of Readme of `deepspeechandroid` project)_
the `CUDA_CONFIGURATION` boolean is automatically forced off as the
CUDA computing toolkit is not available for Android. _(And of course Nvidia CUDA supporting GPUs for mobile devices duh)_

### Platform
Set the platform accordingly to your target build platform so that
```cmake
$ENV{JAVA_HOME}/include/${PLATFORM}
```
points to the platform specific JNI headers.

####Examples
```cmake
set(PLATFORM linux)
```
```cmake
set(PLATFORM win32)
```
```cmake
set(PLATFORM darwin)
```
...

If your target build configuration is different to the configuration
of the system, change `$ENV{JAVA_HOME}` to `$SET_JAVA_HOME`, a variable
that needs to be set to the java home directory of a JDK for your target platform.

eg.
```cmake
set(SET_JAVA_HOME "/usr/lib/jvm/java-8-oracle")
```

## Placing DeepSpeech binaries

Make sure to place the libdeepspeech library for your target platform
in `${CMAKE_SOURCE_DIR}/libs/${PLATFORM}`.

When building for Desktop devices, the library must be named `libdeepspeech_cpu` or `libdeepspeech_cuda` depending
on whether the binary is built to use the CPU or CUDA.

For android, place the [ABI] specific library inside 
`${CMAKE_SOURCE_DIR}/libs/android/${ANDROID_ABI}`.

The `libdeepspeech` library is built for the following [ABI]s:
* arm64-v8a
* armeabi-v7a
* x86_64

The needed `libdeepspeech` library can  be downloaded from the
[DeepSpeech releases] page.

## Running the build

To build the `libdeepspeech-jni` binary, you have to have CMake installed.
Download CMake from here: [CMake releases]

After the project directory is setup correctly, the `libdeepspeech-jni`
library can be build using the command.

```
cmake build .
```

Your binary will be placed in the CMake build directory.
eg `cmake-build-debug` or `cmake-build-release`.

[ABI]: https://developer.android.com/ndk/guides/abis
[CMake releases]: https://cmake.org/download/
[DeepSpeech releases]: https://github.com/mozilla/DeepSpeech/releases