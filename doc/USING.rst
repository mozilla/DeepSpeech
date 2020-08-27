.. _usage-docs:

Using a Pre-trained Model
=========================

Inference using a DeepSpeech pre-trained model can be done with a client/language binding package. We have four clients/language bindings in this repository, listed below, and also a few community-maintained clients/language bindings in other repositories, listed `further down in this README <#third-party-bindings>`_.

* :ref:`The C API <c-usage>`.
* :ref:`The Python package/language binding <py-usage>`
* :ref:`The Node.JS package/language binding <nodejs-usage>`
* :ref:`The command-line client <cli-usage>`
* :github:`The .NET client/language binding <native_client/dotnet/README.rst>`

.. _runtime-deps:

Running ``deepspeech`` might, see below, require some runtime dependencies to be already installed on your system:

* ``sox`` - The Python and Node.JS clients use SoX to resample files to 16kHz.
* ``libgomp1`` - libsox (statically linked into the clients) depends on OpenMP. Some people have had to install this manually.
* ``libstdc++`` - Standard C++ Library implementation. Some people have had to install this manually.
* ``libpthread`` - On Linux, some people have had to install libpthread manually. On Ubuntu, ``libpthread`` is part of the ``libpthread-stubs0-dev`` package.  
* ``Redistribuable Visual C++ 2015 Update 3 (64-bits)`` - On Windows, it might be required to ensure this is installed. Please `download from Microsoft <https://www.microsoft.com/download/details.aspx?id=53587>`_.

Please refer to your system's documentation on how to install these dependencies.

.. _cuda-deps:

CUDA dependency
^^^^^^^^^^^^^^^

The GPU capable builds (Python, NodeJS, C++, etc) depend on CUDA 10.1 and CuDNN v7.6.

Getting the pre-trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use the pre-trained English model for performing speech-to-text, you can download it (along with other important inference material) from the DeepSpeech `releases page <https://github.com/mozilla/DeepSpeech/releases>`_. Alternatively, you can run the following command to download the model files in your current directory:

.. code-block:: bash

   wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.pbmm
   wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.scorer

There are several pre-trained model files available in official releases. Files ending in ``.pbmm`` are compatible with clients and language bindings built against the standard TensorFlow runtime. Usually these packages are simply called ``deepspeech``. These files are also compatible with CUDA enabled clients and language bindings. These packages are usually called ``deepspeech-gpu``. Files ending in ``.tflite`` are compatible with clients and language bindings built against the `TensorFlow Lite runtime <https://www.tensorflow.org/lite/>`_. These models are optimized for size and performance in low power devices. On desktop platforms, the compatible packages are called ``deepspeech-tflite``. On Android and Raspberry Pi, we only publish TensorFlow Lite enabled packages, and they are simply called ``deepspeech``. You can see a full list of supported platforms and which TensorFlow runtime is supported at :ref:`supported-platforms-inference`.

+--------------------+---------------------+---------------------+
| Package/Model type | .pbmm               | .tflite             |
+====================+=====================+=====================+
| deepspeech         | Depends on platform | Depends on platform |
+--------------------+---------------------+---------------------+
| deepspeech-gpu     | ✅                  | ❌                  |
+--------------------+---------------------+---------------------+
| deepspeech-tflite  | ❌                  | ✅                  |
+--------------------+---------------------+---------------------+

Finally, the pre-trained model files also include files ending in ``.scorer``. These are external scorers (language models) that are used at inference time in conjunction with an acoustic model (``.pbmm`` or ``.tflite`` file) to produce transcriptions. We also provide further documentation on :ref:`the decoding process <decoder-docs>` and :ref:`how scorers are generated <scorer-scripts>`.

Important considerations on model inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The release notes include detailed information on how the released models were trained/constructed. Important considerations for users include the characteristics of the training data used and whether they match your intended use case. For acoustic models, an important characteristic is the demographic distribution of speakers. For external scorers, the texts should be similar to those of the expected use case. If the data used for training the models does not align with your intended use case, it may be necessary to adapt or train new models in order to get good accuracy in your transcription results.

The process for training an acoustic model is described in :ref:`training-docs`. In particular, fine tuning a release model using your own data can be a good way to leverage relatively smaller amounts of data that would not be sufficient for training a new model from scratch. See the :ref:`fine tuning and transfer learning sections <training-fine-tuning>` for more information. :ref:`Data augmentation <training-data-augmentation>` can also be a good way to increase the value of smaller training sets.

Creating your own external scorer from text data is another way that you can adapt the model to your specific needs. The process and tools used to generate an external scorer package are described in :ref:`scorer-scripts` and an overview of how the external scorer is used by DeepSpeech to perform inference is available in :ref:`decoder-docs`. Generating a smaller scorer from a single purpose text dataset is a quick process and can bring significant accuracy improvements, specially for more constrained, limited vocabulary applications.

Model compatibility
^^^^^^^^^^^^^^^^^^^

DeepSpeech models are versioned to keep you from trying to use an incompatible graph with a newer client after a breaking change was made to the code. If you get an error saying your model file version is too old for the client, you should either upgrade to a newer model release, re-export your model from the checkpoint using a newer version of the code, or downgrade your client if you need to use the old model and can't re-export it.

.. _py-usage:

Using the Python package
^^^^^^^^^^^^^^^^^^^^^^^^

Pre-built binaries which can be used for performing inference with a trained model can be installed with ``pip3``. You can then use the ``deepspeech`` binary to do speech-to-text on an audio file:

For the Python bindings, it is highly recommended that you perform the installation within a Python 3.5 or later virtual environment. You can find more information about those in `this documentation <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

We will continue under the assumption that you already have your system properly setup to create new virtual environments.

Create a DeepSpeech virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In creating a virtual environment you will create a directory containing a ``python3`` binary and everything needed to run deepspeech. You can use whatever directory you want. For the purpose of the documentation, we will rely on ``$HOME/tmp/deepspeech-venv``. You can create it using this command:

.. code-block::

   $ virtualenv -p python3 $HOME/tmp/deepspeech-venv/

Once this command completes successfully, the environment will be ready to be activated.

Activating the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each time you need to work with DeepSpeech, you have to *activate* this virtual environment. This is done with this simple command:

.. code-block::

   $ source $HOME/tmp/deepspeech-venv/bin/activate

Installing DeepSpeech Python bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your environment has been set-up and loaded, you can use ``pip3`` to manage packages locally. On a fresh setup of the ``virtualenv``\ , you will have to install the DeepSpeech wheel. You can check if ``deepspeech`` is already installed with ``pip3 list``.

To perform the installation, just use ``pip3`` as such:

.. code-block::

   $ pip3 install deepspeech

If ``deepspeech`` is already installed, you can update it as such:

.. code-block::

   $ pip3 install --upgrade deepspeech

Alternatively, if you have a supported NVIDIA GPU on Linux, you can install the GPU specific package as follows:

.. code-block::

   $ pip3 install deepspeech-gpu

See the `release notes <https://github.com/mozilla/DeepSpeech/releases>`_ to find which GPUs are supported. Please ensure you have the required `CUDA dependency <#cuda-dependency>`_.

You can update ``deepspeech-gpu`` as follows:

.. code-block::

   $ pip3 install --upgrade deepspeech-gpu

In both cases, ``pip3`` should take care of installing all the required dependencies. After installation has finished, you should be able to call ``deepspeech`` from the command-line.

Note: the following command assumes you `downloaded the pre-trained model <#getting-the-pre-trained-model>`_.

.. code-block:: bash

   deepspeech --model deepspeech-0.7.4-models.pbmm --scorer deepspeech-0.7.4-models.scorer --audio my_audio_file.wav

The ``--scorer`` argument is optional, and represents an external language model to be used when transcribing the audio.

See :ref:`the Python client <py-api-example>` for an example of how to use the package programatically.

.. _nodejs-usage:

Using the Node.JS / Electron.JS package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can download the JS bindings using ``npm``\ :

.. code-block:: bash

   npm install deepspeech

Please note that as of now, we support:
 - Node.JS versions 4 to 13.
 - Electron.JS versions 1.6 to 7.1

TypeScript support is also provided.

Alternatively, if you're using Linux and have a supported NVIDIA GPU, you can install the GPU specific package as follows:

.. code-block:: bash

   npm install deepspeech-gpu

See the `release notes <https://github.com/mozilla/DeepSpeech/releases>`_ to find which GPUs are supported. Please ensure you have the required `CUDA dependency <#cuda-dependency>`_.

See the :ref:`TypeScript client <js-api-example>` for an example of how to use the bindings programatically.

.. _cli-usage:

Using the command-line client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download the pre-built binaries for the ``deepspeech`` command-line (compiled C++) client, use ``util/taskcluster.py``\ :

.. code-block:: bash

   python3 util/taskcluster.py --target .

or if you're on macOS:

.. code-block:: bash

   python3 util/taskcluster.py --arch osx --target .

also, if you need some binaries different than current master, like ``v0.2.0-alpha.6``\ , you can use ``--branch``\ :

.. code-block:: bash

   python3 util/taskcluster.py --branch "v0.2.0-alpha.6" --target "."

The script ``taskcluster.py`` will download ``native_client.tar.xz`` (which includes the ``deepspeech`` binary and associated libraries) and extract it into the current folder. Also, ``taskcluster.py`` will download binaries for Linux/x86_64 by default, but you can override that behavior with the ``--arch`` parameter. See the help info with ``python util/taskcluster.py -h`` for more details. Specific branches of DeepSpeech or TensorFlow can be specified as well.

Alternatively you may manually download the ``native_client.tar.xz`` from the [releases](https://github.com/mozilla/DeepSpeech/releases).

Note: the following command assumes you `downloaded the pre-trained model <#getting-the-pre-trained-model>`_.

.. code-block:: bash

   ./deepspeech --model deepspeech-0.7.4-models.pbmm --scorer deepspeech-0.7.4-models.scorer --audio audio_input.wav

See the help output with ``./deepspeech -h`` for more details.

Installing bindings from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If pre-built binaries aren't available for your system, you'll need to install them from scratch. Follow the :github:`native client build and installation instructions <native_client/README.rst>`.

Dockerfile for building from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide ``Dockerfile.build`` to automatically build ``libdeepspeech.so``, the C++ native client, Python bindings, and KenLM.
You need to generate the Dockerfile from the template using:

.. code-block:: bash

   make Dockerfile.build

If you want to specify a different DeepSpeech repository / branch, you can pass ``DEEPSPEECH_REPO`` or ``DEEPSPEECH_SHA`` parameters:

.. code-block:: bash

   make Dockerfile.build DEEPSPEECH_REPO=git://your/fork DEEPSPEECH_SHA=origin/your-branch

Third party bindings
^^^^^^^^^^^^^^^^^^^^

In addition to the bindings above, third party developers have started to provide bindings to other languages:


* `Asticode <https://github.com/asticode>`_ provides `Golang <https://golang.org>`_ bindings in its `go-astideepspeech <https://github.com/asticode/go-astideepspeech>`_ repo.
* `RustAudio <https://github.com/RustAudio>`_ provide a `Rust <https://www.rust-lang.org>`_ binding, the installation and use of which is described in their `deepspeech-rs <https://github.com/RustAudio/deepspeech-rs>`_ repo.
* `stes <https://github.com/stes>`_ provides preliminary `PKGBUILDs <https://wiki.archlinux.org/index.php/PKGBUILD>`_ to install the client and python bindings on `Arch Linux <https://www.archlinux.org/>`_ in the `arch-deepspeech <https://github.com/stes/arch-deepspeech>`_ repo.
* `gst-deepspeech <https://github.com/Elleo/gst-deepspeech>`_ provides a `GStreamer <https://gstreamer.freedesktop.org/>`_ plugin which can be used from any language with GStreamer bindings.
* `thecodrr <https://github.com/thecodrr>`_ provides `Vlang <https://vlang.io>`_ bindings. The installation and use of which is described in their `vspeech <https://github.com/thecodrr/vspeech>`_ repo.
* `eagledot <https://gitlab.com/eagledot>`_ provides `NIM-lang <https://nim-lang.org/>`_ bindings. The installation and use of which is described in their `nim-deepspeech <https://gitlab.com/eagledot/nim-deepspeech>`_ repo.
