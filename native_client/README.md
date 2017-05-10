# DeepSpeech native client

A native client for running queries on an exported DeepSpeech model.

## Requirements

* [TensorFlow source](https://www.tensorflow.org/install/install_sources)
* [libsox](https://sourceforge.net/projects/sox/)

## Preparation

Create a symbolic link in the TensorFlow checkout to the deepspeech `native_client` directory.

```
cd tensorflow
ln -s ../DeepSpeech/native_client ./
```

## Building

Before building the DeepSpeech client libraries, you will need to prepare your environment to configure and build TensorFlow. Follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation'.

Then you can build the Tensorflow and DeepSpeech libraries.

```
bazel build -c opt --copt=-march=native --copt=-mtune=native --copt=-O3 //tensorflow:libtensorflow.so //native_client:deepspeech
```

Finally, you can change to the `native_client` directory and use the `Makefile`. By default, the `Makefile` will assume there is a TensorFlow checkout in a directory above the DeepSpeech checkout. If that is not the case, set the environment variable `TFDIR` to point to the right directory.

```
cd ../DeepSpeech/native_client
make deepspeech
```

## Running

The client can be run via the `Makefile`. The client will accept audio of any format your installation of SoX supports.

```
ARGS="/path/to/output_graph.pb /path/to/audio/file.ogg" make run
```

## Python bindings

Included are a set of generated Python bindings. After following the above build instructions, these can be installed by executing the following commands (or equivalent on your system):

```
PREFIX=/usr/local sudo make install
make bindings
sudo pip install dist/deepspeech*
```

It is assumed that `$PREFIX/lib` exists in the library path, otherwise you may need to alter your environment. The API mirrors the C++ API and is demonstrated in [client.py](client.py). Refer to [deepspeech.h](deepspeech.h) for documentation.
