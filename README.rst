Project DeepSpeech
==================


.. image:: https://readthedocs.org/projects/deepspeech/badge/?version=latest
   :target: http://deepspeech.readthedocs.io/?badge=latest
   :alt: Documentation


.. image:: https://community-tc.services.mozilla.com/api/github/v1/repository/mozilla/DeepSpeech/master/badge.svg
   :target: https://community-tc.services.mozilla.com/api/github/v1/repository/mozilla/DeepSpeech/master/latest
   :alt: Task Status


DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on `Baidu's Deep Speech research paper <https://arxiv.org/abs/1412.5567>`_. Project DeepSpeech uses Google's `TensorFlow <https://www.tensorflow.org/>`_ to make the implementation easier.

To install and use deepspeech all you have to do is:

.. code-block:: bash

   # Create and activate a virtualenv
   virtualenv -p python3 $HOME/tmp/deepspeech-venv/
   source $HOME/tmp/deepspeech-venv/bin/activate

   # Install DeepSpeech
   pip3 install deepspeech

   # Download pre-trained English model and extract
   curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz
   tar xvf deepspeech-0.5.1-models.tar.gz

   # Download example audio files
   curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/audio-0.5.1.tar.gz
   tar xvf audio-0.5.1.tar.gz

   # Transcribe an audio file
   deepspeech --model deepspeech-0.5.1-models/output_graph.pbmm --lm deepspeech-0.5.1-models/lm.binary --trie deepspeech-0.5.1-models/trie --audio audio/2830-3980-0043.wav

A pre-trained English model is available for use and can be downloaded using `the instructions below <USING.rst#using-a-pre-trained-model>`_. Currently, only 16-bit, 16 kHz, mono-channel WAVE audio files are supported in the Python client. A package with some example audio files is available for download in our `release notes <https://github.com/mozilla/DeepSpeech/releases/latest>`_.

Quicker inference can be performed using a supported NVIDIA GPU on Linux. See the `release notes <https://github.com/mozilla/DeepSpeech/releases/latest>`_ to find which GPUs are supported. To run ``deepspeech`` on a GPU, install the GPU specific package:

.. code-block:: bash

   # Create and activate a virtualenv
   virtualenv -p python3 $HOME/tmp/deepspeech-gpu-venv/
   source $HOME/tmp/deepspeech-gpu-venv/bin/activate

   # Install DeepSpeech CUDA enabled package
   pip3 install deepspeech-gpu

   # Transcribe an audio file.
   deepspeech --model deepspeech-0.5.1-models/output_graph.pbmm --lm deepspeech-0.5.1-models/lm.binary --trie deepspeech-0.5.1-models/trie --audio audio/2830-3980-0043.wav

Please ensure you have the required `CUDA dependencies <USING.rst#cuda-dependency>`_.

See the output of ``deepspeech -h`` for more information on the use of ``deepspeech``. (If you experience problems running ``deepspeech``\ , please check `required runtime dependencies <native_client/README.rst#required-dependencies>`_\ ).

----

**Table of Contents**

* `Trying out DeepSpeech with examples <examples/EXAMPLES.rst>`_

  * `Microphone VAD streaming  <examples/mic_vad_streaming/README.rst>`_
  
  * `FFMPEG VAD streaming  <examples/ffmpeg_vad_streaming/README.rst>`_
  
  * `Net framework  <examples/net_framework/README.rst>`_
  
  * `Nodejs wav  <examples/nodejs_wav/README.rst>`_
  
  * `VAD transcriber  <examples/vad_transcriber/README.rst>`_
  
* `Using a Pre-trained Model <USING.rst#using-a-pre-trained-model>`_

  * `CUDA dependency <USING.rst#cuda-dependency>`_
  * `Getting the pre-trained model <USING.rst#getting-the-pre-trained-model>`_
  * `Model compatibility <USING.rst#model-compatibility>`_
  * `Using the Python package <USING.rst#using-the-python-package>`_
  * `Using the Node.JS package <USING.rst#using-the-nodejs-package>`_
  * `Using the Command Line client <USING.rst#using-the-command-line-client>`_
  * `Installing bindings from source <USING.rst#installing-bindings-from-source>`_
  * `Third party bindings <USING.rst#third-party-bindings>`_

* `Training your own Model <TRAINING.rst#training-your-own-model>`_

  * `Prerequisites for training a model <TRAINING.rst#prerequisites-for-training-a-model>`_
  * `Getting the training code <TRAINING.rst#getting-the-training-code>`_
  * `Installing Python dependencies <TRAINING.rst#installing-python-dependencies>`_
  * `Recommendations <TRAINING.rst#recommendations>`_
  * `Common Voice training data <TRAINING.rst#common-voice-training-data>`_
  * `Training a model <TRAINING.rst#training-a-model>`_
  * `Checkpointing <TRAINING.rst#checkpointing>`_
  * `Exporting a model for inference <TRAINING.rst#exporting-a-model-for-inference>`_
  * `Exporting a model for TFLite <TRAINING.rst#exporting-a-model-for-tflite>`_
  * `Making a mmap-able model for inference <TRAINING.rst#making-a-mmap-able-model-for-inference>`_
  * `Continuing training from a release model <TRAINING.rst#continuing-training-from-a-release-model>`_
  * `Training with Augmentation <TRAINING.rst#training-with-augmentation>`_

* `Contribution guidelines <CONTRIBUTING.rst>`_
* `Contact/Getting Help <SUPPORT.rst>`_
