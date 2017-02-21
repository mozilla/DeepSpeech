# DeepSpeech client

A client for running queries on an exported DeepSpeech model.

## Requirements

* [Tensorflow Serving](https://tensorflow.github.io/serving/setup)

## Preparation

Create a symbolic link in the Tensorflow Serving checkout to the deepspeech client directory.

```
cd serving
ln -s ../DeepSpeech/serving_client ./deepspeech_client
```

## Building

Binaries will be built automatically when required, but can be built in advance.

To build the Tensorflow Server, execute the following command:

```
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
```

Then you can build the DeepSpeech client.

```
bazel build -c opt //deepspeech_client:deepspeech_client
```

## Running

Start a server running an exported DeepSpeech model.

```
bazel run -c opt -- //tensorflow_serving/model_servers:tensorflow_model_server --port=9000 --model_name=deepspeech --model_base_path=/path/to/deepspeech/export
```

Now run the client.

```
bazel run -c opt -- //deepspeech_client:deepspeech_client --server=localhost:9000 --file=/path/to/audio.wav
```
