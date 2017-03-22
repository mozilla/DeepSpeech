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

Before building the TensorFlow stand-alone library, you will need to prepare your environment to configure and build TensorFlow. Follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation'.

To build the TensorFlow library, execute the following command:

```
bazel build -c opt //tensorflow:libtensorflow.so
```

Then you can build the DeepSpeech native library.

```
bazel build -c opt //native_client:*
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
