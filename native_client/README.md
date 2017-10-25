# DeepSpeech native client, language bindings and custom decoder

This folder contains a native client for running queries on an exported DeepSpeech model, bindings for Python and Node.JS for using an exported DeepSpeech model programatically, and a CTC beam search decoder implementation that scores beams using a language model, needed for training a DeepSpeech model. We provide pre-built binaries for Linux and macOS.

## Installation

To download the pre-built binaries, use `util/tc.py`:

```
python util/tc.py --target /path/to/destination/folder
```

This will download and extract `native_client.tar.xz` which includes the deepspeech binary and associated libraries as well as the custom decoder OP. `tc.py` will download binaries for the architecture of the host by default, but you can override that behavior with the `--arch` parameter. See the help info with `python util/tc.py -h` for more details.

If you want the CUDA capable version of the binaries, use `--arch gpu`. Note that for now we don't publish CUDA-capable macOS binaries.

If you're looking train a model, you now have a `libctc_decoder_with_kenlm.so` file that you can pass to the `--decoder_library_path` parameter of `DeepSpeech.py`.

## Installing the language bindings

`native_client.tar.xz` doesn't include the language bindings by default. For that you can use the `--artifact` parameter to download a specific language binding file.

For Python bindings, use `--artifact file_name`, where `file_name` is the appropriate file for your Python version and platform. The names of the available artifacts can be found on the listing page: [Linux](https://tools.taskcluster.net/index/artifacts/project.deepspeech.deepspeech.native_client.master/cpu) or [macOS](https://tools.taskcluster.net/index/artifacts/project.deepspeech.deepspeech.native_client.master/osx).

For example, for Python 2.7 bindings on Linux, you can do `python util/tc.py --target /destination --artifact deepspeech-0.0.1-cp27-cp27mu-linux_x86_64.whl`.

For Node.JS bindings, use `--arttifact deepspeech-0.0.1.tgz`.

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

Before building the DeepSpeech client libraries, you will need to prepare your environment to configure and build TensorFlow. Follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation'.

Then you can build the Tensorflow and DeepSpeech libraries.

```
bazel build -c opt --copt=-O3 //tensorflow:libtensorflow_cc.so //native_client:deepspeech //native_client:deepspeech_utils //native_client:ctc_decoder_with_kenlm //native_client:generate_trie
```

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
ARGS="/path/to/output_graph.pb /path/to/audio/file.ogg" make run
```

## Python bindings

Included are a set of generated Python bindings. After following the above build and installation instructions, these can be installed by executing the following commands (or equivalent on your system):

```
cd native_client
make bindings
sudo pip install dist/deepspeech*
```

The API mirrors the C++ API and is demonstrated in [client.py](client.py). Refer to [deepspeech.h](deepspeech.h) for documentation.

## Node.JS bindings

After following the above build and installation instructions, the Node.JS bindings can be built:

```
cd native_client/javascript
make
make npm-pack
```

This will create the package `deepspeech-0.0.1.tgz` in the current folder.
