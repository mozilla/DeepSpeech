# Project DeepSpeech

Project DeepSpeech is an open source Speech-To-Text engine that uses a model trained by machine learning techniques, based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) project to facilitate implementation.

# Training LibriSpeech
I've added this section as a clear example of how to train and use LibriSpeech in DeepSpeech.  The original documentation follows after this section.

## Setting up LibriSpeech
If you already have the files copy them to `data/`, if not `import_librivox.py` below will download them in any case.
```
# these are the LibriSpeech files you need
dev-clean.tar.gz
dev-other.tar.gz
test-clean.tar.gz
test-other.tar.gz
train-clean-100.tar.gz
train-clean-360.tar.gz
train-other-500.tar.gz
```
Setup a virtualenv as you might normally, I prefer using the Python 3.6, but 3.5 will do.
```
cd DeepSpeech
virtualenv -p /usr/bin/python3.6 .env
. .env/bin/activate
pip install -r requirements.txt
# if you're using a GPU, afterwards just execute
pip install tensorflow-gpu==1.1.0
```
Run the converter on the LibriSpeech set (this takes a long time as there are ~ 290K files)
I've modified `bin/import-librivox.sh` to use ffmpeg as the sox transformer doesn't seem to support flac on Ubuntu 16.04.
```
./bin/import_librivox.py data/
```

## Training with LibriSpeech
I've modified the checkpoint tensorflow directory default to go into ./data/ckpt
```
mkdir data/ckpt
nohup ./bin/run-librivox.sh &
```
After which you'll probably want to run tensorboard to keep an eye on it.
Tensorboard's default port is 6006 on your machine.
you'll want to keep an eye on the `global_set` SCALAR to view progress in your browser.
```
tensorboard --logdir=/path/to/repo/data/ckpt
# you can now browse hostname:6006 on this machine - wait 10 minutes for the first checkpoint
```

## Converting a checkpoint to a binary graph for use with the native client
In order to use any of the checkpoint files in the data/ckpt folder you'll need to convert them.
This is part of the DeepSpeech.py program, but I've simplified this code and you can execute `export_graph.py` 
to create a new binary graph using the latest checkpoint.
The longer you let the system run, the better your model.  DeepSpeech creates a new model every 10 minutes by default.
I'd recommend several days of training on a single GPU/CPU machine before using any model.
```
./export_graph.py --checkpoint_dir data/ckpt --export_dir /some/folder/for/the/new/graph
```

## Building the native client (and Tensorflow 1.1.0)
The native client requires tensorflow to compile, which in turn requires bazel.  
The following instructions work for Ubuntu 16.04.

### Install bazel
```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
```
### Setup tensorflow version 1.1.0
The DeepSpeech code has been setup to have TensorFlow up one directory from it and puts its own little hooks inside Tensorflow.
I recommend you use all the defaults for `./configure`, from inside the virtualenv you've created above.
So if you're inside the DeepSpeech repo, go:
```
cd ..
git clone git@github.com:tensorflow/tensorflow.git
git checkout v1.1.0
./configure
# link the native client into TF
ln -s ../DeepSpeech/native_client ./
# and finally build TF and the native client
bazel build -c opt --copt=-march=native --copt=-mtune=native --copt=-O3
```

### Build the native client after setting up Tensorflow
Execute the make file for the native client
```
cd DeepSpeech/native_client
make deepspeech
# install the native client into your Linux system
sudo make install
# update the linkages
sudo ldconfig
```

### Using the native client
Now that you've build the native client you can use it as follows.
First you need to convert your sound file to 16KHz mono using ffmpeg as follows:
```
ffmpeg -i any_sound_file.mp3 -acodec pcm_s16le -ac 1 -ar 16000 deep_speech_input.wav
```
Finally, you can convert your sound file to text (from anywhere in your system since you installed it OS wide),
this requires the previously generated binary protocol-buffer graph-file [See Converting a checkpoint to a binary graph for use with the native client](#converting-a-checkpoint-to-a-binary-graph-for-use-with-the-native-client)
```
# outputs the text to stdout using DeepSpeech
deepspeech /path/to/graph/output_graph.pb /path/to/deep_speech_input.wav
```

# Project DeepSpeech [![Documentation Status](https://readthedocs.org/projects/deepspeech/badge/?version=latest)](http://deepspeech.readthedocs.io/en/latest/?badge=latest)

Project DeepSpeech is an open source Speech-To-Text engine that uses a model trained by machine learning techniques, based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) project to facilitate implementation.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Install](#install)
- [Recommendations](#recommendations)
- [Training a model](#training-a-model)
- [Checkpointing](#checkpointing)
- [Exporting a model for serving](#exporting-a-model-for-serving)
- [Distributed computing across more than one machine](#distributed-computing-across-more-than-one-machine)
- [Documentation](#documentation)
- [Contact/Getting Help](#contactgetting-help)

## Prerequisites

* [Git Large File Storage](https://git-lfs.github.com/)
* [TensorFlow 1.0 or 1.1](https://www.tensorflow.org/install/)
* [SciPy](https://scipy.org/install.html)
* [PyXDG](https://pypi.python.org/pypi/pyxdg)
* [python_speech_features](https://pypi.python.org/pypi/python_speech_features)
* [python sox](https://pypi.python.org/pypi/sox)
* [pandas](https://pypi.python.org/pypi/pandas)


## Install

Manually install [Git Large File Storage](https://git-lfs.github.com/), then open a terminal and run:
```bash
git clone https://github.com/mozilla/DeepSpeech
cd DeepSpeech
pip install -r requirements.txt
```

## Recommendations

If you have a capable (Nvidia, at least 8GB of VRAM) GPU, it is highly recommended to install TensorFlow with GPU support. Training will likely be significantly quicker than using the CPU.

## Training a model

The central (Python) script is `DeepSpeech.py` in the project's root directory. For its list of command line options, you can call:

```bash
$ ./DeepSpeech.py --help
```

To get the output of this in a slightly better formatted way, you can also look up the option definitions top of `DeepSpeech.py`.
For executing pre-configured training scenarios, there is a collection of convenience scripts in the `bin` folder. Most of them are named after the corpora they are configured for. Keep in mind that the other speech corpora are *very large*, on the order of tens of gigabytes, and some aren't free. Downloading and preprocessing them can take a very long time, and training on them without a fast GPU (GTX 10 series recommended) takes even longer. If you experience GPU OOM errors while training, try reducing `batch_size`.

As a simple first example you can open a terminal, change to the directory of the DeepSpeech checkout and run:

```bash
$ ./bin/run-ldc93s1.sh
```

This script will train on a small sample dataset called LDC93S1, which can be overfitted on a GPU in a few minutes for demonstration purposes. From here, you can alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the network parameters.
Feel also free to pass additional (or overriding) `DeepSpeech.py` parameters to these scripts.
Then, just run the script to train the modified network.

Each dataset has a corresponding importer script in `bin/` that can be used to download (if it's freely available) and preprocess the dataset. See `bin/import_librivox.py` for an example of how to import and preprocess a large dataset for training with Deep Speech.

If you've ran the old importers (in `util/importers/`), they could have removed source files that are needed for the new importers to run. In that case, simply remove the extracted folders and let the importer extract and process the dataset from scratch, and things should work.

## Checkpointing

During training of a model so called checkpoints will get stored on disk. This takes place at a configurable time interval. The purpose of checkpoints is to allow interruption (also in case of some unexpected failure) and later continuation of training without loosing hours of training time. Resuming from checkpoints happens automatically by just (re)starting training with the same `--checkpoint_dir` of the former run.

Be aware however that checkpoints are only valid for the same model geometry they had been generated from. In other words: If there are error messages of certain `Tensors` having incompatible dimensions, this is most likely due to an incompatible model change. One usual way out would be to wipe all checkpoint files in the checkpoint directory or changing it before starting the training.

## Exporting a model for serving

If the `--export_dir` parameter is provided, a model will have been exported to this directory during training.
Refer to the corresponding [README.md](native_client/README.md) for information on building and running a client that can use the exported model.

## Distributed computing across more than one machine

DeepSpeech has built-in support for [distributed TensorFlow](https://www.tensorflow.org/deploy/distributed). To get an idea on how this works, you can use the script `bin/run-cluster.sh` for running a cluster with workers just on the local machine.

```bash
$ bin/run-cluster.sh --help
Usage: run-cluster.sh [--help] [--script script] [p:w:g] <arg>*

--help      print this help message
--script    run the provided script instead of DeepSpeech.py
p           number of local parameter servers
w           number of local workers
g           number of local GPUs per worker
<arg>*      remaining parameters will be forwarded to DeepSpeech.py or a provided script

Example usage - The following example will create a local DeepSpeech.py cluster
with 1 parameter server, and 2 workers with 1 GPU each:
$ run-cluster.sh 1:2:1 --epoch 10
```

Be aware that for the help example to be able to run, you need at least two `CUDA` capable GPUs (2 workers times 1 GPU). The script utilizes environment variable `CUDA_VISIBLE_DEVICES` for `DeepSpeech.py` to see only the provided number of GPUs per worker.
The script is meant to be a template for your own distributed computing instrumentation. Just modify the startup code for the different servers (workers and parameter servers) accordingly. You could use SSH or something similar for running them on your remote hosts.

## Documentation

Documentation (incomplete) for the project can be found here: http://deepspeech.readthedocs.io/en/latest/

## Contact/Getting Help

First, check out our existing issues and the [FAQ on the wiki](https://github.com/mozilla/DeepSpeech/wiki) to see if your question is answered there. If it's not, and the question is about the code or the project's goals, feel free to open an issue in the repo. If the question is better suited for the FAQ, the team hangs out in the #machinelearning channel on [Mozilla IRC](https://wiki.mozilla.org/IRC), and people there can try to answer/help.
