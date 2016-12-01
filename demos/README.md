# DeepSpeech demos

A collection of demos for an exported DeepSpeech model.

## Requirements

* [Tensorflow Serving](https://tensorflow.github.io/serving/setup)
* [python-websocket-server](https://github.com/Pithikos/python-websocket-server)
* [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)

## Building

Create a symbolic link in the Tensorflow Serving checkout to the deepspeech demos directory.

```
cd serving
ln -s ../DeepSpeech/demos ./deepspeech_demos
```

If you haven't already, you'll need to build the Tensorflow Server.

```
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
```

Then you can build the DeepSpeech demos binary.

```
bazel build -c opt //deepspeech_demos
```

## Running

Start a server running an exported DeepSpeech model.

```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=deepspeech --model_base_path=/path/to/deepspeech/export
```

Run the demos binary, from the demos directory.

```
/path/to/tensorflow/serving/bazel-bin/deepspeech_demos/deepspeech_demos --server=localhost:9000
```

Now navigate to http://localhost:8080 in a web browser.
