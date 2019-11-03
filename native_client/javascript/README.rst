Project DeepSpeech
==================

[![Task Status](https://community-tc.services.mozilla.com/api/github/v1/repository/mozilla/DeepSpeech/master/badge.svg)](https://community-tc.services.mozilla.com/api/github/v1/repository/mozilla/DeepSpeech/master/latest)

DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on `Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow <https://www.tensorflow.org/>`_. Project DeepSpeech uses Google's `TensorFlow <https://www.tensorflow.org/>`_ to make the implementation easier.

!`Usage <images/usage.gif>`_

Pre-built binaries for performing inference with a trained model can be installed with `pip3`. Proper setup using a virtual environment is recommended, and you can find that documentation `below <#using-the-python-package>`_.

A pre-trained English model is available for use and can be downloaded using `the instructions below <#getting-the-pre-trained-model>`_. Currently, only 16-bit, 16 kHz, mono-channel WAVE audio files are supported in the Python client.

Once everything is installed, you can then use the `deepspeech` binary to do speech-to-text on short (approximately 5-second long) audio files as such:

```bash

pip3 install deepspeech

deepspeech --model models/output*graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my*audio_file.wav

```

Alternatively, quicker inference can be performed using a supported NVIDIA GPU on Linux. See the `release notes <https://github.com/mozilla/DeepSpeech/releases>`_ to find which GPUs are supported. To run `deepspeech` on a GPU, install the GPU specific package:

```bash

pip3 install deepspeech-gpu

deepspeech --model models/output*graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my*audio_file.wav

```

Please ensure you have the required `CUDA dependency <#cuda-dependency>`_.

See the output of `deepspeech -h` for more information on the use of `deepspeech`. (If you experience problems running `deepspeech`, please check `required runtime dependencies <native*client/README.rst#required-dependencies>`*).

**Table of Contents**

- `Prerequisites <#prerequisites>`_
- `Getting the code <#getting-the-code>`_
- `Using a Pre-trained Model <#using-a-pre-trained-model>`_
  - `CUDA dependency <#cuda-dependency>`_

  - `Getting the pre-trained model <#getting-the-pre-trained-model>`_

  - `Model compatibility <#model-compatibility>`_

  - `Using the Python package <#using-the-python-package>`_

  - `Using the Node.JS package <#using-the-nodejs-package>`_

  - `Using the Command Line client <#using-the-command-line-client>`_

  - `Installing bindings from source <#installing-bindings-from-source>`_

  - `Third party bindings <#third-party-bindings>`_
- `Training your own Model <#training-your-own-model>`_
  - `Installing training prerequisites <#installing-training-prerequisites>`_

  - `Recommendations <#recommendations>`_

  - `Common Voice training data <#common-voice-training-data>`_

  - `Training a model <#training-a-model>`_

  - `Checkpointing <#checkpointing>`_

  - `Exporting a model for inference <#exporting-a-model-for-inference>`_

  - `Exporting a model for TFLite <#exporting-a-model-for-tflite>`_

  - `Making a mmap-able model for inference <#making-a-mmap-able-model-for-inference>`_

  - `Continuing training from a release model <#continuing-training-from-a-release-model>`_
- `Contribution guidelines <#contribution-guidelines>`_
- `Contact/Getting Help <#contactgetting-help>`_

# Prerequisites
===============

* `Python 3.6 <https://www.python.org/>`_

* `Git Large File Storage <https://git-lfs.github.com/>`_

* Mac or Linux environment

* Go to `build README <examples/net*framework/README.rst>`* to start building DeepSpeech for Windows from source.

# Getting the code
==================

Install `Git Large File Storage <https://git-lfs.github.com/>`_ either manually or through a package-manager if available on your system. Then clone the DeepSpeech repository normally:

```bash

git clone https://github.com/mozilla/DeepSpeech

```


# Using a Pre-trained Model
===========================

There are three ways to use DeepSpeech inference:

- `The Python package <#using-the-python-package>`_
- `The Node.JS package <#using-the-nodejs-package>`_
- `The Command-Line client <#using-the-command-line-client>`_

Running `deepspeech` might require some runtime dependencies to be already installed on your system. Regardless of which bindings you are using, you will need the following:

* libsox2

* libstdc++6

* libgomp1

* libpthread

Please refer to your system's documentation on how to install these dependencies.


## CUDA dependency
==================

The GPU capable builds (Python, NodeJS, C++, etc) depend on the same CUDA runtime as upstream TensorFlow. Currently with TensorFlow 1.13 it depends on CUDA 10.0 and CuDNN v7.5.

## Getting the pre-trained model
================================

If you want to use the pre-trained English model for performing speech-to-text, you can download it (along with other important inference material) from the DeepSpeech `releases page <https://github.com/mozilla/DeepSpeech/releases>`_. Alternatively, you can run the following command to download and unzip the model files in your current directory:

```bash

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz

tar xvfz deepspeech-0.5.1-models.tar.gz

```

## Model compatibility
======================

DeepSpeech models are versioned to keep you from trying to use an incompatible graph with a newer client after a breaking change was made to the code. If you get an error saying your model file version is too old for the client, you should either upgrade to a newer model release, re-export your model from the checkpoint using a newer version of the code, or downgrade your client if you need to use the old model and can't re-export it.

## Using the Python package
===========================

Pre-built binaries which can be used for performing inference with a trained model can be installed with `pip3`. You can then use the `deepspeech` binary to do speech-to-text on an audio file:

For the Python bindings, it is highly recommended that you perform the installation within a Python 3.5 or later virtual environment. You can find more information about those in `this documentation <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

We will continue under the assumption that you already have your system properly setup to create new virtual environments.

### Create a DeepSpeech virtual environment
===========================================

In creating a virtual environment you will create a directory containing a `python3` binary and everything needed to run deepspeech. You can use whatever directory you want. For the purpose of the documentation, we will rely on `$HOME/tmp/deepspeech-venv`. You can create it using this command:

```

$ virtualenv -p python3 $HOME/tmp/deepspeech-venv/

```

Once this command completes successfully, the environment will be ready to be activated.

### Activating the environment
==============================

Each time you need to work with DeepSpeech, you have to *activate* this virtual environment. This is done with this simple command:

```

$ source $HOME/tmp/deepspeech-venv/bin/activate

```

### Installing DeepSpeech Python bindings
=========================================

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

See the `release notes](https://github.com/mozilla/DeepSpeech/releases) to find which GPUs are supported. Please ensure you have the required [CUDA dependency <#cuda-dependency>`* to find which GPUs are supported. Please ensure you have the required `CUDA dependency <#cuda-dependency>`*.

You can update `deepspeech-gpu` as follows:

```

$ pip3 install --upgrade deepspeech-gpu

```

In both cases, `pip3` should take care of installing all the required dependencies. After installation has finished, you should be able to call `deepspeech` from the command-line.


Note: the following command assumes you `downloaded the pre-trained model <#getting-the-pre-trained-model>`_.

```bash

deepspeech --model models/output*graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my*audio_file.wav

```

The arguments `--lm` and `--trie` are optional, and represent a language model.

See `client.py <native*client/python/client.py>`* for an example of how to use the package programatically.

## Using the Node.JS package
============================

You can download the Node.JS bindings using `npm`:

```bash

npm install deepspeech

```

Please note that as of now, we only support Node.JS versions 4, 5 and 6. Once `SWIG has support <https://github.com/swig/swig/pull/968>`_ we can build for newer versions.

Alternatively, if you're using Linux and have a supported NVIDIA GPU, you can install the GPU specific package as follows:

```bash

npm install deepspeech-gpu

```

See the `release notes](https://github.com/mozilla/DeepSpeech/releases) to find which GPUs are supported. Please ensure you have the required [CUDA dependency <#cuda-dependency>`* to find which GPUs are supported. Please ensure you have the required `CUDA dependency <#cuda-dependency>`*.

See `client.js](native*client/javascript/client.js) for an example of how to use the bindings. Or download the [wav example <examples/nodejs*wav>`* for an example of how to use the bindings. Or download the `wav example <examples/nodejs*wav>`_.


## Using the Command-Line client
================================

To download the pre-built binaries for the `deepspeech` command-line (compiled C++) client, use `util/taskcluster.py`:

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

The script `taskcluster.py` will download `native*client.tar.xz` (which includes the `deepspeech` binary and associated libraries) and extract it into the current folder. Also, `taskcluster.py` will download binaries for Linux/x86*64 by default, but you can override that behavior with the `--arch` parameter. See the help info with `python util/taskcluster.py -h` for more details. Specific branches of DeepSpeech or TensorFlow can be specified as well.

Note: the following command assumes you `downloaded the pre-trained model <#getting-the-pre-trained-model>`_.

```bash

./deepspeech --model models/output*graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio audio*input.wav

```

See the help output with `./deepspeech -h` and the `native client README <native*client/README.rst>`* for more details.

## Installing bindings from source
==================================

If pre-built binaries aren't available for your system, you'll need to install them from scratch. Follow these ``native*client` installation instructions <native*client/README.rst>`_.

## Third party bindings
=======================

In addition to the bindings above, third party developers have started to provide bindings to other languages:

* `Asticode](https://github.com/asticode) provides [Golang](https://golang.org) bindings in its [go-astideepspeech <https://github.com/asticode/go-astideepspeech>`_ provides `Golang](https://golang.org) bindings in its [go-astideepspeech <https://github.com/asticode/go-astideepspeech>`_ bindings in its `go-astideepspeech <https://github.com/asticode/go-astideepspeech>`_ repo.

* `RustAudio](https://github.com/RustAudio) provide a [Rust](https://www.rust-lang.org) binding, the installation and use of which is described in their [deepspeech-rs <https://github.com/RustAudio/deepspeech-rs>`_ provide a `Rust](https://www.rust-lang.org) binding, the installation and use of which is described in their [deepspeech-rs <https://github.com/RustAudio/deepspeech-rs>`_ binding, the installation and use of which is described in their `deepspeech-rs <https://github.com/RustAudio/deepspeech-rs>`_ repo.

* `stes](https://github.com/stes) provides preliminary [PKGBUILDs](https://wiki.archlinux.org/index.php/PKGBUILD) to install the client and python bindings on [Arch Linux](https://www.archlinux.org/) in the [arch-deepspeech <https://github.com/stes/arch-deepspeech>`_ provides preliminary `PKGBUILDs](https://wiki.archlinux.org/index.php/PKGBUILD) to install the client and python bindings on [Arch Linux](https://www.archlinux.org/) in the [arch-deepspeech <https://github.com/stes/arch-deepspeech>`_ to install the client and python bindings on `Arch Linux](https://www.archlinux.org/) in the [arch-deepspeech <https://github.com/stes/arch-deepspeech>`_ in the `arch-deepspeech <https://github.com/stes/arch-deepspeech>`_ repo.

* `gst-deepspeech](https://github.com/Elleo/gst-deepspeech) provides a [GStreamer <https://gstreamer.freedesktop.org/>`_ provides a `GStreamer <https://gstreamer.freedesktop.org/>`_ plugin which can be used from any language with GStreamer bindings.

# Training Your Own Model
=========================

## Installing Training Prerequisites
====================================

Install the required dependencies using `pip3`:

```bash

cd DeepSpeech

pip3 install -r requirements.txt

```

You'll also need to install the `ds*ctcdecoder` Python package. `ds*ctcdecoder` is required for decoding the outputs of the `deepspeech` acoustic model into text. You can use `util/taskcluster.py` with the `--decoder` flag to get a URL to a binary of the decoder package appropriate for your platform and Python version:

```bash

pip3 install $(python3 util/taskcluster.py --decoder)

```

This command will download and install the `ds*ctcdecoder` package. If you prefer building the binaries from source, see the `native*client README file <native*client/README.rst>`*. You can override the platform with `--arch` if you want the package for ARM7 (`--arch arm`) or ARM64 (`--arch arm64`).

## Recommendations
==================

If you have a capable (NVIDIA, at least 8GB of VRAM) GPU, it is highly recommended to install TensorFlow with GPU support. Training will be significantly faster than using the CPU. To enable GPU support, you can do:

```bash

pip3 uninstall tensorflow

pip3 install 'tensorflow-gpu==1.13.1'

```

Please ensure you have the required `CUDA dependency <#cuda-dependency>`_.

It has been reported for some people failure at training:

```

tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.

	 [[{{node tower\_0/conv1d/Conv2D}}]]

```

Setting the `TF*FORCE*GPU*ALLOW*GROWTH` environment variable to `true` seems to help in such cases.

## Common Voice training data
=============================

The Common Voice corpus consists of voice samples that were donated through Mozilla's `Common Voice <https://voice.mozilla.org/>`_ Initiative.

You can download individual CommonVoice v2.0 language data sets from `here <https://voice.mozilla.org/data>`_.

After extraction of such a data set, you'll find the following contents:

 - the `*.tsv` files output by CorporaCreator for the downloaded language

 - the mp3 audio files they reference in a `clips` sub-directory.

For bringing this data into a form that DeepSpeech understands, you have to run the CommonVoice v2.0 importer (`bin/import_cv2.py`):

```bash

bin/import*cv2.py --filter*alphabet path/to/some/alphabet.txt /path/to/extracted/language/archive

```

Providing a filter alphabet is optional. It will exclude all samples whose transcripts contain characters not in the specified alphabet. 

Running the importer with `-h` will show you some additional options.

Once the import is done, the `clips` sub-directory will contain for each required `.mp3` an additional `.wav` file.

It will also add the following `.csv` files:

- `clips/train.csv`
- `clips/dev.csv`
- `clips/test.csv`

All entries in these CSV files refer to their samples by absolute paths. So moving this sub-directory would require another import or tweaking the CSV files accordingly.

To use Common Voice data during training, validation and testing, you pass (comma separated combinations of) their filenames into `--train*files`, `--dev*files`, `--test_files` parameters of `DeepSpeech.py`.

If, for example, Common Voice language `en` was extracted to `../data/CV/en/`, `DeepSpeech.py` could be called like this:

```bash

./DeepSpeech.py --train*files ../data/CV/en/clips/train.csv --dev*files ../data/CV/en/clips/dev.csv --test_files ../data/CV/en/clips/test.csv

```

## Training a model
===================

The central (Python) script is `DeepSpeech.py` in the project's root directory. For its list of command line options, you can call:

```bash

./DeepSpeech.py --helpfull

```

To get the output of this in a slightly better-formatted way, you can also look up the option definitions top `DeepSpeech.py`.

For executing pre-configured training scenarios, there is a collection of convenience scripts in the `bin` folder. Most of them are named after the corpora they are configured for. Keep in mind that the other speech corpora are *very large*, on the order of tens of gigabytes, and some aren't free. Downloading and preprocessing them can take a very long time, and training on them without a fast GPU (GTX 10 series recommended) takes even longer.

**If you experience GPU OOM errors while training, try reducing the batch size with the `--train*batch*size`, `--dev*batch*size` and `--test*batch*size` parameters.**

As a simple first example you can open a terminal, change to the directory of the DeepSpeech checkout and run:

```bash

./bin/run-ldc93s1.sh

```

This script will train on a small sample dataset called LDC93S1, which can be overfitted on a GPU in a few minutes for demonstration purposes. From here, you can alter any variables with regards to what dataset is used, how many training iterations are run and the default values of the network parameters.

Feel also free to pass additional (or overriding) `DeepSpeech.py` parameters to these scripts. Then, just run the script to train the modified network.

Each dataset has a corresponding importer script in `bin/` that can be used to download (if it's freely available) and preprocess the dataset. See `bin/import_librivox.py` for an example of how to import and preprocess a large dataset for training with DeepSpeech.

If you've run the old importers (in `util/importers/`), they could have removed source files that are needed for the new importers to run. In that case, simply remove the extracted folders and let the importer extract and process the dataset from scratch, and things should work.

## Checkpointing
================

During training of a model so-called checkpoints will get stored on disk. This takes place at a configurable time interval. The purpose of checkpoints is to allow interruption (also in the case of some unexpected failure) and later continuation of training without losing hours of training time. Resuming from checkpoints happens automatically by just (re)starting training with the same `--checkpoint_dir` of the former run.

Be aware however that checkpoints are only valid for the same model geometry they had been generated from. In other words: If there are error messages of certain `Tensors` having incompatible dimensions, this is most likely due to an incompatible model change. One usual way out would be to wipe all checkpoint files in the checkpoint directory or changing it before starting the training.

## Exporting a model for inference
==================================

If the `--export_dir` parameter is provided, a model will have been exported to this directory during training.

Refer to the corresponding `README.rst <native*client/README.rst>`* for information on building and running a client that can use the exported model.

## Exporting a model for TFLite
===============================

If you want to experiment with the TF Lite engine, you need to export a model that is compatible with it, then use the `--export*tflite` flags. If you already have a trained model, you can re-export it for TFLite by running `DeepSpeech.py` again and specifying the same `checkpoint*dir` that you used for training, as well as passing `--export*tflite --export*dir /model/export/destination`.

## Making a mmap-able model for inference
=========================================

The `output_graph.pb` model file generated in the above step will be loaded in memory to be dealt with when running inference.

This will result in extra loading time and memory consumption. One way to avoid this is to directly read data from the disk.

TensorFlow has tooling to achieve this: it requires building the target `//tensorflow/contrib/util:convert*graphdef*memmapped*format` (binaries are produced by our TaskCluster for some systems including Linux/amd64 and macOS/amd64), use `util/taskcluster.py` tool to download, specifying `tensorflow` as a source and `convert*graphdef*memmapped*format` as artifact.

Producing a mmap-able model is as simple as:

```

$ convert*graphdef*memmapped*format --in*graph=output*graph.pb --out*graph=output_graph.pbmm

```

Upon sucessfull run, it should report about conversion of a non-zero number of nodes. If it reports converting `0` nodes, something is wrong: make sure your model is a frozen one, and that you have not applied any incompatible changes (this includes `quantize_weights`).

## Continuing training from a release model
===========================================

If you'd like to use one of the pre-trained models released by Mozilla to bootstrap your training process (transfer learning, fine tuning), you can do so by using the `--checkpoint_dir` flag in `DeepSpeech.py`. Specify the path where you downloaded the checkpoint from the release, and training will resume from the pre-trained model.

For example, if you want to fine tune the entire graph using your own data in `my-train.csv`, `my-dev.csv` and `my-test.csv`, for three epochs, you can something like the following, tuning the hyperparameters as needed:

```bash

mkdir fine*tuning*checkpoints

python3 DeepSpeech.py --n*hidden 2048 --checkpoint*dir path/to/checkpoint/folder --epochs 3 --train*files my-train.csv --dev*files my-dev.csv --test*files my*dev.csv --learning_rate 0.0001

```

Note: the released models were trained with `--n_hidden 2048`, so you need to use that same value when initializing from the release models.

# Contribution guidelines
=========================

This repository is governed by Mozilla's code of conduct and etiquette guidelines. For more details, please read the `Mozilla Community Participation Guidelines <https://www.mozilla.org/about/governance/policies/participation/>`_.

Before making a Pull Request, check your changes for basic mistakes and style problems by using a linter. We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the changed code, you can use the follow command:

```bash

pip install pylint cardboardlint

cardboardlinter --refspec master

```

This will compare the code against master and run the linter on all the changes. We plan to introduce more linter checks (e.g. for C++) in the future. To run it automatically as a git pre-commit hook, do the following:

```bash

cat <<\EOF > .git/hooks/pre-commit
!/bin/bash
==========

if [ ! -x "$(command -v cardboardlinter)" ]; then

	exit 0

fi

First, stash index and work dir, keeping only the
=================================================
to-be-committed changes in the working directory.
=================================================

echo "Stashing working tree changes..." 1>&2

old_stash=$(git rev-parse -q --verify refs/stash)

git stash save -q --keep-index

new_stash=$(git rev-parse -q --verify refs/stash)

If there were no changes (e.g., `--amend` or `--allow-empty`)
=============================================================
then nothing was stashed, and we should skip everything,
========================================================
including the tests themselves.  (Presumably the tests passed
=============================================================
on the previous commit, so there is no need to re-run them.)
============================================================

if [ "$old*stash" = "$new*stash" ]; then

	echo "No changes, skipping lint." 1>&2

	exit 0

fi

Run tests
=========

cardboardlinter --refspec HEAD -n auto

status=$?

Restore changes
===============

echo "Restoring working tree changes..." 1>&2

git reset --hard -q && git stash apply --index -q && git stash drop -q

Exit with status from test-run: nonzero prevents commit
=======================================================

exit $status

EOF

chmod +x .git/hooks/pre-commit

```

This will run the linters on just the changes made in your commit.

# Contact/Getting Help
======================

There are several ways to contact us or to get help:

1. `**FAQ**](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) - We have a list of common questions, and their answers, in our [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions). When just getting started, it's best to first check the [FAQ <https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions>`_ - We have a list of common questions, and their answers, in our `FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions). When just getting started, it's best to first check the [FAQ <https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions>`_. When just getting started, it's best to first check the `FAQ <https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions>`_ to see if your question is addressed.

2. `**Discourse Forums**](https://discourse.mozilla.org/c/deep-speech) - If your question is not addressed in the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions), the [Discourse Forums](https://discourse.mozilla.org/c/deep-speech) is the next place to look. They contain conversations on [General Topics](https://discourse.mozilla.org/t/general-topics/21075), [Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development <https://discourse.mozilla.org/t/deep-speech-development/21077>`_ - If your question is not addressed in the `FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions), the [Discourse Forums](https://discourse.mozilla.org/c/deep-speech) is the next place to look. They contain conversations on [General Topics](https://discourse.mozilla.org/t/general-topics/21075), [Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development <https://discourse.mozilla.org/t/deep-speech-development/21077>`_, the `Discourse Forums](https://discourse.mozilla.org/c/deep-speech) is the next place to look. They contain conversations on [General Topics](https://discourse.mozilla.org/t/general-topics/21075), [Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development <https://discourse.mozilla.org/t/deep-speech-development/21077>`_ is the next place to look. They contain conversations on `General Topics](https://discourse.mozilla.org/t/general-topics/21075), [Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development <https://discourse.mozilla.org/t/deep-speech-development/21077>`_, `Using Deep Speech](https://discourse.mozilla.org/t/using-deep-speech/21076/4), and [Deep Speech Development <https://discourse.mozilla.org/t/deep-speech-development/21077>`_, and `Deep Speech Development <https://discourse.mozilla.org/t/deep-speech-development/21077>`_.

3. `**IRC**](https://wiki.mozilla.org/IRC) - If your question is not addressed by either the [FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) or [Discourse Forums](https://discourse.mozilla.org/c/deep-speech), you can contact us on the `#machinelearning` channel on [Mozilla IRC <https://wiki.mozilla.org/IRC>`_ - If your question is not addressed by either the `FAQ](https://github.com/mozilla/DeepSpeech/wiki#frequently-asked-questions) or [Discourse Forums](https://discourse.mozilla.org/c/deep-speech), you can contact us on the `#machinelearning` channel on [Mozilla IRC <https://wiki.mozilla.org/IRC>`_ or `Discourse Forums](https://discourse.mozilla.org/c/deep-speech), you can contact us on the `#machinelearning` channel on [Mozilla IRC <https://wiki.mozilla.org/IRC>`_, you can contact us on the `#machinelearning` channel on `Mozilla IRC <https://wiki.mozilla.org/IRC>`_; people there can try to answer/help

4. `**Issues** <https://github.com/mozilla/deepspeech/issues>`_ - Finally, if all else fails, you can open an issue in our repo.

