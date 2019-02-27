# DeepSpeech native client, language bindings, and custom decoder

This folder contains the following:

1. A native client for running queries on an exported DeepSpeech model
2. Python and Node.JS bindings for using an exported DeepSpeech model programatically
3. A CTC beam search decoder which uses a language model (N.B - the decoder is also required for training DeepSpeech)

We provide pre-built binaries for Linux and macOS.

## Installation

To download the pre-built binaries, use `util/taskcluster.py`:

```
python util/taskcluster.py --target /path/to/destination/folder
```

If you need binaries which are different than current master (e.g. `v0.2.0-alpha.6`) you can use the `--branch` flag:

```bash
python3 util/taskcluster.py --branch "v0.2.0-alpha.6"
```

`util/taskcluster.py` will download and extract `native_client.tar.xz`.  `native_client.tar.xz` includes (1) the `deepspeech` binary and (2) associated libraries. `taskcluster.py` will download binaries for the architecture of the host by default, but you can override that behavior with the `--arch` parameter. See `python util/taskcluster.py -h` for more details.

If you want the CUDA capable version of the binaries, use `--arch gpu`. Note that for now we don't publish CUDA-capable macOS binaries.

## Required Dependencies

Running inference might require some runtime dependencies to be already installed on your system. Those should be the same, whatever the bindings you are using:
* libsox2
* libstdc++6
* libgomp1
* libpthread

Please refer to your system's documentation on how to install those dependencies.

## Installing the language bindings

### Python bindings

For the Python bindings, you can use `pip`:

```
pip install deepspeech
```

Check the [main README](../README.md) for more details about setup and virtual environment use.

### Node.JS bindings

For Node.JS bindings, use `npm install deepspeech` to install it. Please note that as of now, we only support Node.JS versions 4, 5 and 6. Once [SWIG has support](https://github.com/swig/swig/pull/968) we can build for newer versions.

Check the [main README](../README.md) for more details.

## Build Requirements

If you'd like to build the binaries yourself, you'll need the following pre-requisites downloaded and installed:

* [TensorFlow requirements](https://www.tensorflow.org/install/install_sources)
* [TensorFlow `r1.12` sources](https://github.com/mozilla/tensorflow/tree/r1.12)
* [libsox](https://sourceforge.net/projects/sox/)

It is required to use our fork of TensorFlow since it includes fixes for common problems encountered when building the native client files.

If you'd like to build the language bindings or the decoder package, you'll also need:

* [SWIG](http://www.swig.org/)
* [node-pre-gyp](https://github.com/mapbox/node-pre-gyp) (for Node.JS bindings only)

## Preparation

Create a symbolic link in your TensorFlow checkout to the DeepSpeech `native_client` directory. If your DeepSpeech and TensorFlow checkouts are side by side in the same directory, do:

```
cd tensorflow
ln -s ../DeepSpeech/native_client ./
```

## Building

Before building the DeepSpeech client libraries, you will need to prepare your environment to configure and build TensorFlow.

Preferably, checkout the version of `tensorflow` which is currently supported by DeepSpeech (see requirements.txt), and use the `bazel` version recommended by TensorFlow for that version.
Then, follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of ["Configure the Build"](https://www.tensorflow.org/install/source#configure_the_build).

After that, you can build the Tensorflow and DeepSpeech libraries using the following command.

```
bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie
```

If your build target requires extra flags, add them. For example `--config=cuda` if you want a CUDA build. Note that the generated binaries will show up under `bazel-bin/native_client/` (e.g., including `generate_trie` in case the `//native_client:generate_trie` option was present).

Finally, you can change to the `native_client` directory and use the `Makefile`. By default, the `Makefile` will assume there is a TensorFlow checkout in a directory above the DeepSpeech checkout. If that is not the case, set the environment variable `TFDIR` to point to the right directory.

```
cd ../DeepSpeech/native_client
make deepspeech
```

### Cross-building for RPi3 ARMv7 and LePotato ARM64

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

## Installing

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

To build the `ds_ctcdecoder` package, you'll need the general requirements listed above (in particular SWIG). The command below builds the bindings using 8 processes for compilation. Adjust the parameter accordingly for more or less parallelism.

```
cd native_client/ctcdecode
make bindings NUM_PROCESSES=8
pip install dist/*.whl
```


## Running

The client can be run via the `Makefile`. The client will accept audio of any format your installation of SoX supports.

```
ARGS="--model /path/to/output_graph.pbmm --alphabet /path/to/alphabet.txt --audio /path/to/audio/file.wav" make run
```
