# Project DeepSpeech

[![Documentation Status](https://readthedocs.org/projects/deepspeech/badge/?version=master)](http://deepspeech.readthedocs.io/?badge=master)
[![Task Status](https://github.taskcluster.net/v1/repository/mozilla/DeepSpeech/master/badge.svg)](https://github.taskcluster.net/v1/repository/mozilla/DeepSpeech/master/latest)

Project DeepSpeech is an open source Speech-To-Text engine. It uses a model trained by machine learning techniques, based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) project to make the implementation easier.

![Usage](images/usage.gif)

Pre-built binaries that can be used for performing inference with a trained model can be installed with `pip`. Proper setup using virtual environment is recommended and you can find that documented [below](#using-the-python-package). You can then use the `deepspeech` binary to do speech-to-text on an audio file:

```bash
pip install deepspeech
deepspeech output_model.pb my_audio_file.wav alphabet.txt
```

See the output of `deepspeech -h` for more information.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Getting the code](#getting-the-code)
- [Using the model](#using-the-model)
  - [Using the Python package](#using-the-python-package)
  - [Using the command line client](#using-the-command-line-client)
  - [Using the Node.JS package](#using-the-nodejs-package)
- [Training](#training)
  - [Recommendations](#recommendations)
  - [Training a model](#training-a-model)
  - [Checkpointing](#checkpointing)
  - [Exporting a model for inference](#exporting-a-model-for-inference)
  - [Distributed computing across more than one machine](#distributed-training-across-more-than-one-machine)
- [Documentation](#documentation)
- [Contact/Getting Help](#contactgetting-help)

## Prerequisites

* [Python 2.7](https://www.python.org/)
* [Git Large File Storage](https://git-lfs.github.com/)

## Getting the code

Manually install [Git Large File Storage](https://git-lfs.github.com/), then clone the repository normally:

```bash
git clone https://github.com/mozilla/DeepSpeech
```

## Using the model

If all you want to do is use an already trained model for doing speech-to-text, you can grab one of our pre-built binaries. You can use a command-line binary, a Python package, or a Node.JS package.

### Using the Python package

Pre-built binaries that can be used for performing inference with a trained model can be installed with `pip`. You can then use the `deepspeech` binary to do speech-to-text on an audio file:

For the Python bindings, it is highly recommended that you perform the installation within a virtual environment. You can find more information about those in [this documentation](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
We will continue under the assumption that you already have your system properly setup to create new virtual environments.

#### Create a DeepSpeech virtual environment

In creating a virtual environment you will create a directory containing a `python` binary and everything needed to run deepspeech. You can use whatever directory you want. For the purpose of the documentation, we will rely on `$HOME/tmp/deepspeech-venv`. You can create it using this command:

```
$ virtualenv $HOME/tmp/deepspeech-venv/
```

Once this command completes successfully, the environment will be ready to be activated.

#### Activating the environment

Each time you need to work with DeepSpeech, you have to *activate*, *load* this virtual environment. This is done with this simple command:

```
$ source $HOME/tmp/deepspeech-venv/bin/activate
```

#### Installing DeepSpeech Python bindings

Once your environment has been setup and loaded, you can use `pip` to manage packages locally. On a fresh setup of the virtualenv, you will have to install the DeepSpeech wheel. You can check if it is already installed by taking a look at the output of `pip list`. To perform the installation, just issue:

```
$ pip install deepspeech
```

If it is already installed, you can also update it:
```
$ pip install --upgrade deepspeech
```

In both cases, it should take care of intalling all the required dependencies. Once it is done, you should be able to call the sample binary using `deepspeech` on your command-line.

```bash
deepspeech output_model.pb my_audio_file.wav alphabet.txt lm.binary trie
```

See [client.py](native_client/python/client.py) for an example of how to use the package programatically.

### Using the command-line client

To download the pre-built binaries, use `util/taskcluster.py`:

```bash
python util/taskcluster.py --target .
```

or if you're on macOS:

```bash
python util/taskcluster.py --arch osx --target .
```

This will download `native_client.tar.xz` which includes the deepspeech binary and associated libraries, and extract it into the current folder. `taskcluster.py` will download binaries for Linux/x86_64 by default, but you can override that behavior with the `--arch` parameter. See the help info with `python util/taskcluster.py -h` for more details.

```bash
./deepspeech model.pb audio_input.wav alphabet.txt lm.binary trie
```

See the help output with `./deepspeech -h` and the [native client README](native_client/README.md) for more details.

### Using the Node.JS package

You can download the Node.JS bindings using `npm`:

```bash
npm install deepspeech
```

Alternatively, if you're using Linux and have a supported NVIDIA GPU (See the release notes to find which GPU's are supported.), you can install the GPU specific package as follows:

```bash
npm install deepspeech-gpu
```

See [client.js](native_client/javascript/client.js) for an example of how to use the bindings.

### Installing bindings from source

If pre-built binaries aren't available for your system, you'll need to install them from scratch. Follow [these instructions](native_client/README.md).

## Training

### Installing prerequisites for training

Install the required dendencies using pip:

```bash
cd DeepSpeech
python util/taskcluster.py --target /tmp --source tensorflow --artifact tensorflow_warpctc-1.3.0rc0-cp27-cp27mu-linux_x86_64.whl
pip install /tmp/tensorflow_warpctc-1.3.0rc0-cp27-cp27mu-linux_x86_64.whl
pip install -r requirements.txt
```

You'll also need to download `native_client.tar.xz` or build the native client files yourself to get the custom TensorFlow OP needed for decoding the outputs of the neural network. You can use `util/taskcluster.py` to download the files for your architecture:

```bash
python util/taskcluster.py --target .
```

This will download the native client files for the x86_64 architecture without CUDA support, and extract them into the current folder. If you prefer building the binaries from source, see the [native_client README file](native_client/README.md). We also have binaries with CUDA enabled ("--arch gpu") and for ARM7 ("--arch arm").

### Recommendations

If you have a capable (Nvidia, at least 8GB of VRAM) GPU, it is highly recommended to install TensorFlow with GPU support. Training will likely be significantly quicker than using the CPU. To enable GPU support, you can do:

```bash
pip uninstall tensorflow
python util/taskcluster.py --target /tmp --source tensorflow --arch gpu --artifact tensorflow_gpu_warpctc-1.3.0rc0-cp27-cp27mu-linux_x86_64.whl
pip install /tmp/tensorflow_gpu_warpctc-1.3.0rc0-cp27-cp27mu-linux_x86_64.whl
```

### Common Voice training data

The Common Voice corpus consists of voice samples that were donated through [Common Voice](https://voice.mozilla.org/).
For automatically downloading and importing the corpus into a given directory, you can call:

```bash
bin/import_cv.py path/to/a/directory
```

Please be aware that this requires at least 70GB of free disk space and quite some time to conclude. 
As this process creates a huge number of small files, using an SSD drive is highly recommended.
If the import script gets interrupted, it will try to continue from where it stopped the next time you run it. 
Unfortunately, there are some cases where it will need to start over. 
Once the import is done, the directory will contain a bunch of CSV files.

The following files are official user-validated sets for training, validating and testing:

- `cv-valid-train.csv`
- `cv-valid-dev.csv`
- `cv-valid-test.csv`

The following files are the non-validated unofficial sets for training, validating and testing:

- `cv-other-train.csv`
- `cv-other-dev.csv`
- `cv-other-test.csv`

`cv-invalid.csv` contains all samples that users flagged as invalid.

A sub-directory called `cv_corpus_{version}` contains the mp3 and wav files that were extracted from an archive named `cv_corpus_{version}.tar.gz`.
All entries in the CSV files refer to their samples by absolute paths. So moving this sub-directory would require another import or tweaking the CSV files accordingly.

To use Common Voice data during training, validation and testing, you pass (comma separated combinations of) their filenames into `--train_files`, `--dev_files`, `--test_files` parameters of `DeepSpeech.py`. 
If, for example, Common Voice was imported into `../data/CV`, `DeepSpeech.py` could be called like this:

```bash
./DeepSpeech.py --train_files ../data/CV/cv-valid-train.csv,../data/CV/cv-other-train.csv --dev_files ../data/CV/cv-valid-dev.csv --test_files ../data/CV/cv-valid-test.csv
```

### Training a model

The central (Python) script is `DeepSpeech.py` in the project's root directory. For its list of command line options, you can call:

```bash
./DeepSpeech.py --help
```

To get the output of this in a slightly better-formatted way, you can also look up the option definitions top of `DeepSpeech.py`.

For executing pre-configured training scenarios, there is a collection of convenience scripts in the `bin` folder. Most of them are named after the corpora they are configured for. Keep in mind that the other speech corpora are *very large*, on the order of tens of gigabytes, and some aren't free. Downloading and preprocessing them can take a very long time, and training on them without a fast GPU (GTX 10 series recommended) takes even longer.

**If you experience GPU OOM errors while training, try reducing the batch size with the `--train_batch_size`, `--dev_batch_size` and `--test_batch_size` parameters.**

As a simple first example you can open a terminal, change to the directory of the DeepSpeech checkout and run:

```bash
./bin/run-ldc93s1.sh
```

This script will train on a small sample dataset called LDC93S1, which can be overfitted on a GPU in a few minutes for demonstration purposes. From here, you can alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the network parameters.
Feel also free to pass additional (or overriding) `DeepSpeech.py` parameters to these scripts.
Then, just run the script to train the modified network.

Each dataset has a corresponding importer script in `bin/` that can be used to download (if it's freely available) and preprocess the dataset. See `bin/import_librivox.py` for an example of how to import and preprocess a large dataset for training with Deep Speech.

If you've run the old importers (in `util/importers/`), they could have removed source files that are needed for the new importers to run. In that case, simply remove the extracted folders and let the importer extract and process the dataset from scratch, and things should work.

### Checkpointing

During training of a model so-called checkpoints will get stored on disk. This takes place at a configurable time interval. The purpose of checkpoints is to allow interruption (also in the case of some unexpected failure) and later continuation of training without losing hours of training time. Resuming from checkpoints happens automatically by just (re)starting training with the same `--checkpoint_dir` of the former run.

Be aware however that checkpoints are only valid for the same model geometry they had been generated from. In other words: If there are error messages of certain `Tensors` having incompatible dimensions, this is most likely due to an incompatible model change. One usual way out would be to wipe all checkpoint files in the checkpoint directory or changing it before starting the training.

### Exporting a model for inference

If the `--export_dir` parameter is provided, a model will have been exported to this directory during training.
Refer to the corresponding [README.md](native_client/README.md) for information on building and running a client that can use the exported model.

### Distributed training across more than one machine

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

There are several ways to contact us or to get help:

1. [**FAQ**](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) - We have a list of common questions, and their answers, in our [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions). When just getting started, it's best to first check the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) to see if your question is addressed.

2. [**Discourse Forums**](https://discourse.mozilla.org/c/deep-speech) - If your question is not addressed in the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions), the [Discourse Forums](https://discourse.mozilla.org/c/deep-speech) is the next place to look. They contain conversations on [General Topics](https://discourse.mozilla.org/t/general-topics/21075), [Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development](https://discourse.mozilla.org/t/deep-speech-development/21077).

3. [**IRC**](https://wiki.mozilla.org/IRC) - If your question is not addressed by either the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) or [Discourse Forums](https://discourse.mozilla.org/c/deep-speech), you can contact us on the `#machinelearning` channel on [Mozilla IRC](https://wiki.mozilla.org/IRC); people there can try to answer/help

4. [**Issues**](https://github.com/mozilla/deepspeech/issues) - Finally, if all else fails, you can open an issue in our repo.
