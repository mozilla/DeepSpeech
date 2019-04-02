# DeepSpeech native client, language bindings, and custom decoder

This folder contains the following:

1. A native client for running queries on an exported DeepSpeech model
2. Python and Node.JS bindings for using an exported DeepSpeech model programatically
3. A CTC beam search decoder which uses a language model (N.B - the decoder is also required for training DeepSpeech)

We provide pre-built binaries for Linux and macOS. If you want to build your own, read the [building instructions here](BUILD.md).

## Required Dependencies

Running inference might require some runtime dependencies to be already installed on your system. Those should be the same, whatever the bindings you are using:
* libsox2
* libstdc++6
* libgomp1
* libpthread

Please refer to your system's documentation on how to install those dependencies.

## Installing our Pre-built Binaries

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


## Installing our Pre-built language bindings

### Python bindings

For the Python bindings, you can use `pip`:

```
pip install deepspeech
```

Check the [main README](../README.md) for more details about setup and virtual environment use.

### Node.JS bindings

For Node.JS bindings, use `npm install deepspeech` to install it. Please note that as of now, we only support Node.JS versions 4, 5 and 6. Once [SWIG has support](https://github.com/swig/swig/pull/968) we can build for newer versions.

Check the [main README](../README.md) for more details.

