# Project DeepSpeech

[![Task Status](https://github.taskcluster.net/v1/repository/mozilla/DeepSpeech/master/badge.svg)](https://github.taskcluster.net/v1/repository/mozilla/DeepSpeech/master/latest)

DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) to make the implementation easier.

![Usage](images/usage.gif)

Pre-built binaries for performing inference with a trained model can be installed with `pip3`. Proper setup using a virtual environment is recommended, and you can find that documentation [below](#using-the-python-package).

A pre-trained English model is available for use and can be downloaded using [the instructions below](#getting-the-pre-trained-model). Currently, only 16-bit, 16 kHz, mono-channel WAVE audio files are supported in the Python client.

Once everything is installed, you can then use the `deepspeech` binary to do speech-to-text on short (approximately 5-second long) audio files as such:

```bash
pip3 install deepspeech
deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my_audio_file.wav
```

Alternatively, quicker inference can be performed using a supported NVIDIA GPU on Linux. See the [release notes](https://github.com/mozilla/DeepSpeech/releases) to find which GPUs are supported. To run `deepspeech` on a GPU, install the GPU specific package:

```bash
pip3 install deepspeech-gpu
deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my_audio_file.wav
```

Please ensure you have the required [CUDA dependency](#cuda-dependency).

See the output of `deepspeech -h` for more information on the use of `deepspeech`. (If you experience problems running `deepspeech`, please check [required runtime dependencies](native_client/README.md#required-dependencies)).

**Table of Contents**

- [Prerequisites](#prerequisites)
  - [CUDA dependency](#cuda-dependency)
- [Using a Pre-trained model](#using-the-model)
  - [Getting the pre-trained model](#getting-the-pre-trained-model)
  - [Using the Command Line (C++) client](#using-the-command-line-client)
  - [Using the Python package](#using-the-python-package)
  - [Using the Node.JS package](#using-the-nodejs-package)
  - [Installing bindings from source](#installing-bindings-from-source)
  - [Third party bindings](#third-party-bindings)
- [Training your own Model](#training)
  - [Getting the code](#getting-the-code)
  - [Installing training prerequisites](#installing-prerequisites-for-training)
  - [Recommendations](#recommendations)
  - [Common Voice training data](#common-voice-training-data)
  - [Training a model](#training-a-model)
  - [Checkpointing](#checkpointing)
  - [Exporting a model for inference](#exporting-a-model-for-inference)
  - [Exporting a model for TFLite](#exporting-a-model-for-tflite)
  - [Continuing training from a release model](#continuing-training-from-a-release-model)
- [Contact/Getting Help](#contactgetting-help)

## Prerequisites

* [Python 3.6](https://www.python.org/)
* [Git Large File Storage](https://git-lfs.github.com/)
* Mac or Linux environment
* Go to [build README](examples/net_framework/README.md) to start building DeepSpeech for Windows from source.

## Getting the code

Install [Git Large File Storage](https://git-lfs.github.com/) either manually or through a package-manager if available on your system. Then clone the DeepSpeech repository normally:

```bash
git clone https://github.com/mozilla/DeepSpeech
```

## Getting the pre-trained model

If you want to use the pre-trained English model for performing speech-to-text, you can download it (along with other important inference material) from the DeepSpeech [releases page](https://github.com/mozilla/DeepSpeech/releases). Alternatively, you can run the following command to download and unzip the model files in your current directory:

```bash
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-models.tar.gz
tar xvfz deepspeech-0.4.1-models.tar.gz
```

## Using the model

There are three ways to use DeepSpeech inference:

- [The Python package](#using-the-python-package)
- [The command-line client](#using-the-command-line-client)
- [The Node.JS package](#using-the-nodejs-package)


Running `deepspeech` might require some runtime dependencies to be already installed on your system. Regardless of which bindings you are using, you will need the following:

* libsox2
* libstdc++6
* libgomp1
* libpthread

Please refer to your system's documentation on how to install these dependencies.


### CUDA dependency

The GPU capable builds (Python, NodeJS, C++, etc) depend on the same CUDA runtime as upstream TensorFlow. Make sure you've installed the correct version of CUDA

### Using the Command-Line client

To download the pre-built binaries for the `deepspeech` command-line client, use `util/taskcluster.py`:

```bash
python3 util/taskcluster.py --target .
```

or if you're on macOS:

```bash
python3 util/taskcluster.py --arch osx --target .
```

also, if you need some binaries different than current master, like `v0.2.0-alpha.6`, you can use `--branch`:

```bash
python3 util/taskcluster.py --branch "v0.2.0-alpha.6" --target "."
```

The script `taskcluster.py` will download `native_client.tar.xz` (which includes the `deepspeech` binary and associated libraries) and extract it into the current folder. Also, `taskcluster.py` will download binaries for Linux/x86_64 by default, but you can override that behavior with the `--arch` parameter. See the help info with `python util/taskcluster.py -h` for more details. Specific branches of DeepSpeech or TensorFlow can be specified as well.

Note: the following command assumes you [downloaded the pre-trained model](#getting-the-pre-trained-model).

```bash
./deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio audio_input.wav
```

See the help output with `./deepspeech -h` and the [native client README](native_client/README.md) for more details.


### Using the Python package

Pre-built binaries which can be used for performing inference with a trained model can be installed with `pip3`. You can then use the `deepspeech` binary to do speech-to-text on an audio file:

For the Python bindings, it is highly recommended that you perform the installation within a Python 3.5 or later virtual environment. You can find more information about those in [this documentation](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

We will continue under the assumption that you already have your system properly setup to create new virtual environments.

#### Create a DeepSpeech virtual environment

In creating a virtual environment you will create a directory containing a `python3` binary and everything needed to run deepspeech. You can use whatever directory you want. For the purpose of the documentation, we will rely on `$HOME/tmp/deepspeech-venv`. You can create it using this command:

```
$ virtualenv -p python3 $HOME/tmp/deepspeech-venv/
```

Once this command completes successfully, the environment will be ready to be activated.

#### Activating the environment

Each time you need to work with DeepSpeech, you have to *activate* this virtual environment. This is done with this simple command:

```
$ source $HOME/tmp/deepspeech-venv/bin/activate
```

#### Installing DeepSpeech Python bindings

Once your environment has been set-up and loaded, you can use `pip3` to manage packages locally. On a fresh setup of the `virtualenv`, you will have to install the DeepSpeech wheel. You can check if `deepspeech` is already installed with `pip3 list`.

To perform the installation, just use `pip3` as such:

```
$ pip3 install deepspeech
```

If `deepspeech` is already installed, you can update it as such:

```
$ pip3 install --upgrade deepspeech
```

Alternatively, if you have a supported NVIDIA GPU on Linux, you can install the GPU specific package as follows:

```
$ pip3 install deepspeech-gpu
```

See the [release notes](https://github.com/mozilla/DeepSpeech/releases) to find which GPUs are supported. Please ensure you have the required [CUDA dependency](#cuda-dependency).

You can update `deepspeech-gpu` as follows:

```
$ pip3 install --upgrade deepspeech-gpu
```

In both cases, `pip3` should take care of installing all the required dependencies. After installation has finished, you should be able to call `deepspeech` from the command-line.


Note: the following command assumes you [downloaded the pre-trained model](#getting-the-pre-trained-model).

```bash
deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my_audio_file.wav
```

The arguments `--lm` and `--trie` are optional, and represent a language model.

See [client.py](native_client/python/client.py) for an example of how to use the package programatically.

### Using the Node.JS package

You can download the Node.JS bindings using `npm`:

```bash
npm install deepspeech
```

Please note that as of now, we only support Node.JS versions 4, 5 and 6. Once [SWIG has support](https://github.com/swig/swig/pull/968) we can build for newer versions.

Alternatively, if you're using Linux and have a supported NVIDIA GPU, you can install the GPU specific package as follows:

```bash
npm install deepspeech-gpu
```

See the [release notes](https://github.com/mozilla/DeepSpeech/releases) to find which GPUs are supported. Please ensure you have the required [CUDA dependency](#cuda-dependency).

See [client.js](native_client/javascript/client.js) for an example of how to use the bindings. Or download the [wav example](examples/nodejs_wav).

### Installing bindings from source

If pre-built binaries aren't available for your system, you'll need to install them from scratch. Follow these [`native_client` installation instructions](native_client/README.md).

### Third party bindings

In addition to the bindings above, third party developers have started to provide bindings to other languages:

* [Asticode](https://github.com/asticode) provides [Golang](https://golang.org) bindings in its [go-astideepspeech](https://github.com/asticode/go-astideepspeech) repo.
* [RustAudio](https://github.com/RustAudio) provide a [Rust](https://www.rust-lang.org) binding, the installation and use of which is described in their [deepspeech-rs](https://github.com/RustAudio/deepspeech-rs) repo.
* [stes](https://github.com/stes) provides preliminary [PKGBUILDs](https://wiki.archlinux.org/index.php/PKGBUILD) to install the client and python bindings on [Arch Linux](https://www.archlinux.org/) in the [arch-deepspeech](https://github.com/stes/arch-deepspeech) repo.
* [gst-deepspeech](https://github.com/Elleo/gst-deepspeech) provides a [GStreamer](https://gstreamer.freedesktop.org/) plugin which can be used from any language with GStreamer bindings.

## Training

### Installing prerequisites for training

Install the required dependencies using `pip3`:

```bash
cd DeepSpeech
pip3 install -r requirements.txt
```

You'll also need to install the `ds_ctcdecoder` Python package. `ds_ctcdecoder` is required for decoding the outputs of the `deepspeech` acoustic model into text. You can use `util/taskcluster.py` with the `--decoder` flag to get a URL to a binary of the decoder package appropriate for your platform and Python version:

```bash
pip3 install $(python3 util/taskcluster.py --decoder)
```

This command will download and install the `ds_ctcdecoder` package. If you prefer building the binaries from source, see the [native_client README file](native_client/README.md). You can override the platform with `--arch` if you want the package for ARM7 (`--arch arm`) or ARM64 (`--arch arm64`).

### Recommendations

If you have a capable (NVIDIA, at least 8GB of VRAM) GPU, it is highly recommended to install TensorFlow with GPU support. Training will be significantly faster than using the CPU. To enable GPU support, you can do:

```bash
pip3 uninstall tensorflow
pip3 install 'tensorflow-gpu==1.13.1'
```

Please ensure you have the required [CUDA dependency](#cuda-dependency).

### Common Voice training data

The Common Voice corpus consists of voice samples that were donated through Mozilla's [Common Voice](https://voice.mozilla.org/) Initiative.

We provide an importer (`bin/import_cv2.py`) which automates preparation of the Common Voice (v2.0) corpus as such:

```bash
bin/import_cv2.py /path/to/audio/data_dir /path/to/tsv_dir
```

You should have already downloaded the Common Voice v2.0 data from [here](https://voice.mozilla.org/data). The `import_cv2.py` script assumes as input (1) the audio downloaded from Common Voice v2.0 for a certain language, in addition to (2) the `*.tsv` files output by CorporaCreator (included in Common Voice v2.0 download). As output, the script returns the data and transcripts in a state usable by `DeepSpeech.py` (i.e. `*.csv` and `.WAV` data).

Please be aware that training with the Common Voice corpus archive requires a lot of free disk space and quite some time to conclude. As this process creates a huge number of small files, using an SSD drive is highly recommended. If the import script gets interrupted, it will try to continue from where it stopped the next time you run it. Unfortunately, there are some cases where it will need to start over. Once the import is done, the directory will contain a bunch of CSV files.

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
./DeepSpeech.py --train_files ../data/CV/cv-valid-train.csv --dev_files ../data/CV/cv-valid-dev.csv --test_files ../data/CV/cv-valid-test.csv
```

If you are brave enough, you can also include the `other` dataset, which contains not-yet-validated content:

```bash
./DeepSpeech.py --train_files ../data/CV/cv-valid-train.csv,../data/CV/cv-other-train.csv --dev_files ../data/CV/cv-valid-dev.csv --test_files ../data/CV/cv-valid-test.csv
```

### Training a model

The central (Python) script is `DeepSpeech.py` in the project's root directory. For its list of command line options, you can call:

```bash
./DeepSpeech.py --helpfull
```

To get the output of this in a slightly better-formatted way, you can also look up the option definitions top `DeepSpeech.py`.

For executing pre-configured training scenarios, there is a collection of convenience scripts in the `bin` folder. Most of them are named after the corpora they are configured for. Keep in mind that the other speech corpora are *very large*, on the order of tens of gigabytes, and some aren't free. Downloading and preprocessing them can take a very long time, and training on them without a fast GPU (GTX 10 series recommended) takes even longer.

**If you experience GPU OOM errors while training, try reducing the batch size with the `--train_batch_size`, `--dev_batch_size` and `--test_batch_size` parameters.**

As a simple first example you can open a terminal, change to the directory of the DeepSpeech checkout and run:

```bash
./bin/run-ldc93s1.sh
```

This script will train on a small sample dataset called LDC93S1, which can be overfitted on a GPU in a few minutes for demonstration purposes. From here, you can alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the network parameters.

Feel also free to pass additional (or overriding) `DeepSpeech.py` parameters to these scripts. Then, just run the script to train the modified network.

Each dataset has a corresponding importer script in `bin/` that can be used to download (if it's freely available) and preprocess the dataset. See `bin/import_librivox.py` for an example of how to import and preprocess a large dataset for training with DeepSpeech.

If you've run the old importers (in `util/importers/`), they could have removed source files that are needed for the new importers to run. In that case, simply remove the extracted folders and let the importer extract and process the dataset from scratch, and things should work.

### Checkpointing

During training of a model so-called checkpoints will get stored on disk. This takes place at a configurable time interval. The purpose of checkpoints is to allow interruption (also in the case of some unexpected failure) and later continuation of training without losing hours of training time. Resuming from checkpoints happens automatically by just (re)starting training with the same `--checkpoint_dir` of the former run.

Be aware however that checkpoints are only valid for the same model geometry they had been generated from. In other words: If there are error messages of certain `Tensors` having incompatible dimensions, this is most likely due to an incompatible model change. One usual way out would be to wipe all checkpoint files in the checkpoint directory or changing it before starting the training.

### Exporting a model for inference

If the `--export_dir` parameter is provided, a model will have been exported to this directory during training.
Refer to the corresponding [README.md](native_client/README.md) for information on building and running a client that can use the exported model.

### Exporting a model for TFLite

If you want to experiment with the TF Lite engine, you need to export a model that is compatible with it, then use the `--export_tflite` flag. If you already have a trained model, you can re-export it for TFLite by running `DeepSpeech.py` again and specifying the same `checkpoint_dir` that you used for training, as well as passing `--notrain --notest --export_tflite --export_dir /model/export/destination`.

### Making a mmap-able model for inference

The `output_graph.pb` model file generated in the above step will be loaded in memory to be dealt with when running inference.
This will result in extra loading time and memory consumption. One way to avoid this is to directly read data from the disk.

TensorFlow has tooling to achieve this: it requires building the target `//tensorflow/contrib/util:convert_graphdef_memmapped_format` (binaries are produced by our TaskCluster for some systems including Linux/amd64 and macOS/amd64), use `util/taskcluster.py` tool to download, specifying `tensorflow` as a source and `convert_graphdef_memmapped_format` as artifact.

Producing a mmap-able model is as simple as:

```
$ convert_graphdef_memmapped_format --in_graph=output_graph.pb --out_graph=output_graph.pbmm
```

Upon sucessfull run, it should report about conversion of a non-zero number of nodes. If it reports converting `0` nodes, something is wrong: make sure your model is a frozen one, and that you have not applied any incompatible changes (this includes `quantize_weights`).

### Continuing training from a release model

If you'd like to use one of the pre-trained models released by Mozilla to bootstrap your training process (transfer learning, fine tuning), you can do so by using the `--checkpoint_dir` flag in `DeepSpeech.py`. Specify the path where you downloaded the checkpoint from the release, and training will resume from the pre-trained model.

For example, if you want to fine tune the entire graph using your own data in `my-train.csv`, `my-dev.csv` and `my-test.csv`, for three epochs, you can something like the following, tuning the hyperparameters as needed:

```bash
mkdir fine_tuning_checkpoints
python3 DeepSpeech.py --n_hidden 2048 --checkpoint_dir path/to/checkpoint/folder --epoch -3 --train_files my-train.csv --dev_files my-dev.csv --test_files my_dev.csv --learning_rate 0.0001
```

Note: the released models were trained with `--n_hidden 2048`, so you need to use that same value when initializing from the release models. Note as well the use of a negative epoch count -3 (meaning 3 more epochs) since the checkpoint you're loading from was already trained for several epochs.

## Contact/Getting Help

There are several ways to contact us or to get help:

1. [**FAQ**](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) - We have a list of common questions, and their answers, in our [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions). When just getting started, it's best to first check the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) to see if your question is addressed.

2. [**Discourse Forums**](https://discourse.mozilla.org/c/deep-speech) - If your question is not addressed in the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions), the [Discourse Forums](https://discourse.mozilla.org/c/deep-speech) is the next place to look. They contain conversations on [General Topics](https://discourse.mozilla.org/t/general-topics/21075), [Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development](https://discourse.mozilla.org/t/deep-speech-development/21077).

3. [**IRC**](https://wiki.mozilla.org/IRC) - If your question is not addressed by either the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) or [Discourse Forums](https://discourse.mozilla.org/c/deep-speech), you can contact us on the `#machinelearning` channel on [Mozilla IRC](https://wiki.mozilla.org/IRC); people there can try to answer/help

4. [**Issues**](https://github.com/mozilla/deepspeech/issues) - Finally, if all else fails, you can open an issue in our repo.
