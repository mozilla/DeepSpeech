.. Mozilla Voice STT documentation master file, created by
   sphinx-quickstart on Thu Feb  2 21:20:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Mozilla Voice STT's documentation!
======================================

Mozilla Voice STT is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on `Baidu's Deep Speech research paper <https://arxiv.org/abs/1412.5567>`_. Project Mozilla Voice STT uses Google's `TensorFlow <https://www.tensorflow.org/>`_ to make the implementation easier.

To install and use Mozilla Voice STT all you have to do is:

.. code-block:: bash

   # Create and activate a virtualenv
   virtualenv -p python3 $HOME/tmp/stt-venv/
   source $HOME/tmp/stt-venv/bin/activate

   # Install Mozilla Voice STT
   pip3 install mozilla_voice_stt

   # Download pre-trained English model files
   curl -LO https://github.com/mozilla/STT/releases/download/v0.8.1/deepspeech-0.8.1-models.pbmm
   curl -LO https://github.com/mozilla/STT/releases/download/v0.8.1/deepspeech-0.8.1-models.scorer

   # Download example audio files
   curl -LO https://github.com/mozilla/STT/releases/download/v0.8.1/audio-0.8.1.tar.gz
   tar xvf audio-0.8.1.tar.gz

   # Transcribe an audio file
   mozilla_voice_stt --model deepspeech-0.8.1-models.pbmm --scorer deepspeech-0.8.1-models.scorer --audio audio/2830-3980-0043.wav

A pre-trained English model is available for use and can be downloaded following the instructions in :ref:`the usage docs <usage-docs>`. For the latest release, including pre-trained models and checkpoints, `see the GitHub releases page <https://github.com/mozilla/STT/releases/latest>`_.

Quicker inference can be performed using a supported NVIDIA GPU on Linux. See the `release notes <https://github.com/mozilla/STT/releases/latest>`_ to find which GPUs are supported. To run ``mozilla_voice_stt`` on a GPU, install the GPU specific package:

.. code-block:: bash

   # Create and activate a virtualenv
   virtualenv -p python3 $HOME/tmp/stt-gpu-venv/
   source $HOME/tmp/stt-gpu-venv/bin/activate

   # Install Mozilla Voice STT CUDA enabled package
   pip3 install mozilla_voice_stt_cuda

   # Transcribe an audio file.
   mozilla_voice_stt --model deepspeech-0.8.1-models.pbmm --scorer deepspeech-0.8.1-models.scorer --audio audio/2830-3980-0043.wav

Please ensure you have the required :ref:`CUDA dependencies <cuda-deps>`.

See the output of ``mozilla_voice_stt -h`` for more information on the use of ``mozilla_voice_stt``. (If you experience problems running ``mozilla_voice_stt``, please check :ref:`required runtime dependencies <runtime-deps>`).

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   USING

   TRAINING

   SUPPORTED_PLATFORMS

   BUILDING

.. include:: ../SUPPORT.rst

.. toctree::
   :maxdepth: 2
   :caption: Decoder and scorer

   Decoder

   Scorer

.. toctree::
   :maxdepth: 2
   :caption: Architecture and training

   AcousticModel

   Geometry

   ParallelOptimization

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   Error-Codes

   C-API

   DotNet-API

   Java-API

   NodeJS-API

   Python-API

.. toctree::
   :maxdepth: 2
   :caption: Examples

   C-Examples

   DotNet-Examples

   Java-Examples

   NodeJS-Examples

   Python-Examples

   Contributed-Examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
