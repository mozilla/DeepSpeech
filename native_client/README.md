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

* [TensorFlow requirements](https://www.tensorflow.org/install/install_sources)
* [TensorFlow `r1.12` sources](https://github.com/mozilla/tensorflow/tree/r1.12)
* [libsox](https://sourceforge.net/projects/sox/)

It is required to use our fork of TensorFlow since it includes fixes for common problems encountered when building the native client files.

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
Preferably, checkout the version of tensorflow which is currently supported by DeepSpeech (see requirements.txt), and use the bazel version recommended by TensorFlow for that version.
Then, follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation'.

After that, you can build the Tensorflow and DeepSpeech libraries using the following command.

```
bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie
```

If your build target requires extra flags, add them, like, for example --config=cuda if you do a CUDA build.

Finally, you can change to the `native_client` directory and use the `Makefile`. By default, the `Makefile` will assume there is a TensorFlow checkout in a directory above the DeepSpeech checkout. If that is not the case, set the environment variable `TFDIR` to point to the right directory.

```
cd ../DeepSpeech/native_client
make deepspeech
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
cd native_client/python
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

This will create the package `deepspeech-0.3.0.tgz` in `native_client/javascript`.
