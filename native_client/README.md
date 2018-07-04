# DeepSpeech native client, language bindings and custom decoder

This folder contains a native client for running queries on an exported DeepSpeech model, bindings for Python and Node.JS for using an exported DeepSpeech model programatically, and a CTC beam search decoder implementation that scores beams using a language model, needed for training a DeepSpeech model. We provide pre-built binaries for Linux and macOS.

## Installation

To download the pre-built binaries, use `util/taskcluster.py`:

```
python util/taskcluster.py --target /path/to/destination/folder
```

If you need some binaries different than current master, like `v0.2.0-alpha.6`, you can use `--branch`:
```bash
python3 util/taskcluster.py --branch "v0.2.0-alpha.6"
```

This will download and extract `native_client.tar.xz` which includes the deepspeech binary and associated libraries as well as the custom decoder OP. `taskcluster.py` will download binaries for the architecture of the host by default, but you can override that behavior with the `--arch` parameter. See the help info with `python util/taskcluster.py -h` for more details.

If you want the CUDA capable version of the binaries, use `--arch gpu`. Note that for now we don't publish CUDA-capable macOS binaries.

If you're looking to train a model, you now have a `libctc_decoder_with_kenlm.so` file that you can pass to the `--decoder_library_path` parameter of `DeepSpeech.py`.

## Required Dependencies

Running inference might require some runtime dependencies to be already installed on your system. Those should be the same, whatever the bindings you are using:
* libsox2
* libstdc++6
* libgomp1
* libpthread

Please refer to your system's documentation on how to install those dependencies.

## Installing the language bindings

For the Python bindings, you can use `pip`:

```
pip install deepspeech
```

Check the [main README](../README.md) for more details about setup and virtual environment use.

### Node.JS bindings

For Node.JS bindings, use `npm install deepspeech` to install it. Please note that as of now, we only support Node.JS versions 4, 5 and 6. Once [SWIG has support](https://github.com/swig/swig/pull/968) we can build for newer versions.

Check the [main README](../README.md) for more details.

## Build Requirements

If you'd like to build the binaries yourself, you'll need the following pre-requisites downloaded/installed:

* [TensorFlow source and requirements](https://www.tensorflow.org/install/install_sources)
* [libsox](https://sourceforge.net/projects/sox/)

We recommend using our fork of TensorFlow since it includes fixes for common problems encountered when building the native client files, you can [get it here](https://github.com/mozilla/tensorflow/).

If you'd like to build the language bindings, you'll also need:

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
Preferably, checkout the version of tensorflow which is currently supported by DeepSpeech (see requirements.txt), and use bazel version 0.10.0. 
Then, follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation'.

After that, you can build the Tensorflow and DeepSpeech libraries using the following commands. Please note that the flags for `libctc_decoder_with_kenlm.so` differs a little bit.

```
bazel build -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" //native_client:libctc_decoder_with_kenlm.so
bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:deepspeech_utils //native_client:generate_trie
```

If your build target requires extra flags, add them, like, for example --config=cuda if you do a CUDA build.

Finally, you can change to the `native_client` directory and use the `Makefile`. By default, the `Makefile` will assume there is a TensorFlow checkout in a directory above the DeepSpeech checkout. If that is not the case, set the environment variable `TFDIR` to point to the right directory.

```
cd ../DeepSpeech/native_client
make deepspeech
```

## Building with AOT model

First, please note that this is still experimental. AOT model relies on TensorFlow's [AOT tfcompile](https://www.tensorflow.org/performance/xla/tfcompile) tooling. It takes a protocol buffer file graph as input, and produces a .so library that one can call from C++ code.
To experiment, you will need to build TensorFlow from [github.com/mozilla/tensorflow r1.6 branch](https://github.com/mozilla/tensorflow/tree/r1.6). Follow TensorFlow's documentation for the configuration of your system.
When building, you will have to add some extra parameter and targets.

Bazel defines:
* `--define=DS_NATIVE_MODEL=1`: to toggle AOT support.
* `--define=DS_MODEL_TIMESTEPS=x`: to define how many timesteps you want to handle. Relying on prebuilt model implies we need to use a fixed value for how much audio value we want to use. Timesteps defines that value, and an audio file bigger than this will just be dealt with over several samples. This means there's a compromise between quality and minimum audio segment you want to handle.
* `--define=DS_MODEL_FRAMESIZE=y`: to define your model framesize, this is the second component of your model's input layer shape. Can be extracted using TensorFlow's `summarize_graph` tool.
* `--define=DS_MODEL_FILE=/path/to/graph.pb`: the model you want to use

Bazel targets:
* `//native_client:libdeepspeech_model.so`: to produce `libdeepspeech_model.so`

In the end, the previous example becomes (no change for `libctc_decoder_with_kenlm.so`):

```
bazel build --config=monolithic -c opt --copt=-O3 --copt=-fvisibility=hidden --define=DS_NATIVE_MODEL=1 --define=DS_MODEL_TIMESTEPS=64 --define=DS_MODEL_FRAMESIZE=494 --define=DS_MODEL_FILE=/tmp/model.ldc93s1.pb //native_client:libdeepspeech_model.so //native_client:libdeepspeech.so //native_client:deepspeech_utils //native_client:generate_trie
```

Later, when building either `deepspeech` binaries or bindings, you will have to add some extra variables to your `make` command-line (assuming `TFDIR` points to your TensorFlow's git clone):
```
EXTRA_LIBS="-ldeepspeech_model"
```

## Installing

After building, the library files and binary can optionally be installed to a system path for ease of development. This is also a required step for bindings generation.

```
PREFIX=/usr/local sudo make install
```

It is assumed that `$PREFIX/lib` is a valid library path, otherwise you may need to alter your environment.

## Running

The client can be run via the `Makefile`. The client will accept audio of any format your installation of SoX supports.

```
ARGS="--model /path/to/output_graph.pbmm --alphabet /path/to/alphabet.txt --audio /path/to/audio/file.wav" make run
```

## Python bindings

Included are a set of generated Python bindings. After following the above build and installation instructions, these can be installed by executing the following commands (or equivalent on your system):

```
cd native_client
make bindings
sudo pip install dist/deepspeech*
```

The API mirrors the C++ API and is demonstrated in [client.py](python/client.py). Refer to [deepspeech.h](deepspeech.h) for documentation.

## Node.JS bindings

After following the above build and installation instructions, the Node.JS bindings can be built:

```
cd native_client/javascript
make package
make npm-pack
```

This will create the package `deepspeech-0.1.1.tgz` in `native_client/javascript`.
