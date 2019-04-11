# Building DeepSpeech Binaries

If you'd like to build the DeepSpeech binaries yourself, you'll need the following pre-requisites downloaded and installed:

* [Mozilla's TensorFlow `r1.13` branch](https://github.com/mozilla/tensorflow/tree/r1.13)
* [General TensorFlow requirements](https://www.tensorflow.org/install/install_sources)
* [libsox](https://sourceforge.net/projects/sox/)

It is required to use our fork of TensorFlow since it includes fixes for common problems encountered when building the native client files.

If you'd like to build the language bindings or the decoder package, you'll also need:

* [SWIG](http://www.swig.org/)
* [node-pre-gyp](https://github.com/mapbox/node-pre-gyp) (for Node.JS bindings only)


## Dependencies

If you follow these instructions, you should compile your own binaries of DeepSpeech (built on TensorFlow using Bazel).

For more information on configuring TensorFlow, read the docs up to the end of ["Configure the Build"](https://www.tensorflow.org/install/source#configure_the_build).

### TensorFlow: Clone & Checkout

Clone our fork of TensorFlow and checkout the correct version:

```
git clone https://github.com/mozilla/tensorflow.git
git checkout origin/r1.13
```

### Bazel: Download & Install 

First, [find the version of Bazel](https://www.tensorflow.org/install/source#tested_build_configurations) you need for this TensorFlow release. Next, [download and install the correct version of Bazel](https://docs.bazel.build/versions/master/install.html).

### TensorFlow: Configure with Bazel

After you have installed the correct version of Bazel, configure TensorFlow:

```
cd tensorflow
./configure
```

## Compile DeepSpeech

## Compile `libdeepspeech.so` & `generate_trie`

Within your TensorFlow checkout, create a symbolic link to the DeepSpeech `native_client` directory. Assuming DeepSpeech and TensorFlow checkouts are in the same directory, do:

```
cd tensorflow
ln -s ../DeepSpeech/native_client ./
```

You can now use Bazel to build the main DeepSpeech library, `libdeepspeech.so`, as well as the `generate_trie` binary. Add `--config=cuda` if you want a CUDA build.

```
bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie
```

The generated binaries will be saved to `bazel-bin/native_client/`.

## Compile `libdeepspeech.so` & `generate_trie`

Now, `cd` into the `DeepSpeech/native_client` directory and use the `Makefile` to build everything else (client, Python package, Nodejs package, `ctc_decoder`). Set the environment variable `TFDIR` to point to your TensorFlow checkout.

```
TFDIR=~/tensorflow
cd ../DeepSpeech/native_client
make deepspeech
```




## Installing your own Binaries

After building, the library files and binary can optionally be installed to a system path for ease of development. This is also a required step for bindings generation.

```
PREFIX=/usr/local sudo make install
```

It is assumed that `$PREFIX/lib` is a valid library path, otherwise you may need to alter your environment.

### Python bindings

Included are a set of generated Python bindings. After following the above build and installation instructions, these can be installed by executing the following commands (or equivalent on your system):

```
cd native_client/python
make bindings
pip install dist/deepspeech*
```

The API mirrors the C++ API and is demonstrated in [client.py](python/client.py). Refer to [deepspeech.h](deepspeech.h) for documentation.

### Node.JS bindings

After following the above build and installation instructions, the Node.JS bindings can be built:

```
cd native_client/javascript
make package
make npm-pack
```

This will create the package `deepspeech-VERSION.tgz` in `native_client/javascript`.

### Building the CTC decoder package

To build the `ds_ctcdecoder` package, you'll need the general requirements listed above (in particular SWIG). The command below builds the bindings using eight (8) processes for compilation. Adjust the parameter accordingly for more or less parallelism.

```
cd native_client/ctcdecode
make bindings NUM_PROCESSES=8
pip install dist/*.whl
```

## Cross-building

### RPi3 ARMv7 and LePotato ARM64

We do support cross-compilation. Please refer to our `mozilla/tensorflow` fork, where we define the following `--config` flags:

 - `--config=rpi3` and `--config=rpi3_opt` for Raspbian / ARMv7
 - `--config=rpi3-armv8` and `--config=rpi3-armv8_opt` for ARMBian / ARM64

So your command line for `RPi3` and `ARMv7` should look like:

```
bazel build --config=monolithic --config=rpi3 --config=rpi3_opt -c opt --copt=-O3 --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie
```

And your command line for `LePotato` and `ARM64` should look like:

```
bazel build --config=monolithic --config=rpi3-armv8 --config=rpi3-armv8_opt -c opt --copt=-O3 --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie
```

While we test only on RPi3 Raspbian Stretch and LePotato ARMBian stretch, anything compatible with `armv7-a cortex-a53` or `armv8-a cortex-a53` should be fine.

The `deepspeech` binary can also be cross-built, with `TARGET=rpi3` or `TARGET=rpi3-armv8`. This might require you to setup a system tree using the tool `multistrap` and the multitrap configuration files: `native_client/multistrap_armbian64_stretch.conf` and `native_client/multistrap_raspbian_stretch.conf`.
The path of the system tree can be overridden from the default values defined in `definitions.mk` through the `RASPBIAN` `make` variable.

```
cd ../DeepSpeech/native_client
make TARGET=<system> deepspeech
```

### Android devices

We have preliminary support for Android relying on TensorFlow Lite, with Java and JNI bindinds. For more details on how to experiment with those, please refer to `native_client/java/README.md`.

Please refer to TensorFlow documentation on how to setup the environment to build for Android (SDK and NDK required).

You can build the `libdeepspeech.so` using (ARMv7):

```
bazel build --config=monolithic --config=android --config=android_arm --define=runtime=tflite --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++11 --copt=-D_GLIBCXX_USE_C99 //native_client:libdeepspeech.so
```

Or (ARM64):

```
bazel build --config=monolithic --config=android --config=android_arm64 --define=runtime=tflite --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++11 --copt=-D_GLIBCXX_USE_C99 //native_client:libdeepspeech.so
```

Building the `deepspeech` binary will happen through `ndk-build` (ARMv7):

```
cd ../DeepSpeech/native_client
$ANDROID_NDK_HOME/ndk-build APP_PLATFORM=android-21 APP_BUILD_SCRIPT=$(pwd)/Android.mk NDK_PROJECT_PATH=$(pwd) APP_STL=c++_shared TFDIR=$(pwd)/../../tensorflow/ TARGET_ARCH_ABI=armeabi-v7a
```

And (ARM64):

```
cd ../DeepSpeech/native_client
$ANDROID_NDK_HOME/ndk-build APP_PLATFORM=android-21 APP_BUILD_SCRIPT=$(pwd)/Android.mk NDK_PROJECT_PATH=$(pwd) APP_STL=c++_shared TFDIR=$(pwd)/../../tensorflowx/ TARGET_ARCH_ABI=arm64-v8a 
```

