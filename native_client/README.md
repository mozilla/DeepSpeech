# DeepSpeech native client, language bindings and custom decoder

This folder contains a native client for running queries on an exported DeepSpeech model, bindings for Python and Node.JS for using an exported DeepSpeech model programatically, and a CTC beam search decoder implementation that scores beams using a language model, needed for training a DeepSpeech model. We provide pre-built binaries for Linux and macOS.

## Installation

To download the pre-built binaries, use `util/taskcluster.py`:

```
python util/taskcluster.py --target /path/to/destination/folder
```

This will download and extract `native_client.tar.xz` which includes the deepspeech binary and associated libraries as well as the custom decoder OP. `taskcluster.py` will download binaries for the architecture of the host by default, but you can override that behavior with the `--arch` parameter. See the help info with `python util/taskcluster.py -h` for more details.

If you want the CUDA capable version of the binaries, use `--arch gpu`. Note that for now we don't publish CUDA-capable macOS binaries.

If you're looking to train a model, you now have a `libctc_decoder_with_kenlm.so` file that you can pass to the `--decoder_library_path` parameter of `DeepSpeech.py`.

## Installing the language bindings

`native_client.tar.xz` doesn't include the language bindings by default. For that you can use the `--artifact` parameter to download a specific language binding file.

For Python bindings, use `--artifact file_name`, where `file_name` is the appropriate file for your Python version and platform. The names of the available artifacts can be found on the listing page: [Linux](https://tools.taskcluster.net/index/artifacts/project.deepspeech.deepspeech.native_client.master/cpu) or [macOS](https://tools.taskcluster.net/index/artifacts/project.deepspeech.deepspeech.native_client.master/osx).

For example, for Python 2.7 bindings on Linux, you can do `python util/taskcluster.py --target /destination --artifact deepspeech-0.0.2-cp27-cp27mu-manylinux1_x86_64.whl`.

For Node.JS bindings, use `--artifact deepspeech-0.0.2.tgz`.

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

## Building with AOT model

First, please note that this is still experimental. AOT model relies on TensorFlow's [AOT tfcompile](https://www.tensorflow.org/performance/xla/tfcompile) tooling. It takes a protocol buffer file graph as input, and produces a .so library that one can call from C++ code.
To experiment, you will need to build TensorFlow from [github.com/mozilla/tensorflow master branch](https://github.com/mozilla/tensorflow). Follow TensorFlow's documentation for the configuration of your system.
When building, you will have to add some extra parameter and targets.

Bazel defines:
* `--define=DS_NATIVE_MODEL=1`: to toggle AOT support.
* `--define=DS_MODEL_TIMESTEPS=x`: to define how many timesteps you want to handle. Relying on prebuilt model implies we need to use a fixed value for how much audio value we want to use. Timesteps defines that value, and an audio file bigger than this will just be dealt with over several samples. This means there's a compromise between quality and minimum audio segment you want to handle.
* `--define=DS_MODEL_FRAMESIZE=y`: to define your model framesize, this is the second component of your model's input layer shape. Can be extracted using TensorFlow's `summarize_graph` tool.
* `--define=DS_MODEL_FILE=/path/to/graph.pb`: the model you want to use

Bazel targets:
* `//native_client:deepspeech_model`: to produce `libdeepspeech_model.so`
* `//tensorflow/compiler/aot:runtime `, `//tensorflow/compiler/xla/service/cpu:runtime_matmul`, `//tensorflow/compiler/xla:executable_run_options`

In the end, the previous example becomes:

```
bazel build -c opt --copt=-O3 --define=DS_NATIVE_MODEL=1 --define=DS_MODEL_TIMESTEPS=64 --define=DS_MODEL_FRAMESIZE=494 --define=DS_MODEL_FILE=/tmp/model.ldc93s1.pb //tensorflow:libtensorflow_cc.so //native_client:deepspeech_model //tensorflow/compiler/aot:runtime //tensorflow/compiler/xla/service/cpu:runtime_matmul //tensorflow/compiler/xla:executable_run_options //native_client:deepspeech //native_client:deepspeech_utils //native_client:ctc_decoder_with_kenlm //native_client:generate_trie
```

Later, when building either `deepspeech` binaries or bindings, you will have to add some extra variables to your `make` command-line (assuming `TFDIR` points to your TensorFlow's git clone):
```
EXTRA_LDFLAGS="-L${TFDIR}/bazel-bin/tensorflow/compiler/xla/ -L${TFDIR}/bazel-bin/tensorflow/compiler/aot/ -L${TFDIR}/bazel-bin/tensorflow/compiler/xla/service/cpu/" EXTRA_LIBS="-ldeepspeech_model -lruntime -lexecutable_run_options -lruntime_matmul"
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
make package
make npm-pack
```

This will create the package `deepspeech-0.0.2.tgz` in `native_client/javascript`.
