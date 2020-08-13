Language-Specific Data
======================

This directory contains language-specific data files. Most importantly, you will find here:

1. A list of unique characters for the target language (e.g. English) in ``data/alphabet.txt``. After installing the training code, you can check ``python -m mozilla_voice_stt_training.util.check_characters --help`` for a tool that creates an alphabet file from a list of training CSV files.

2. A script used to generate a binary n-gram language model: ``data/lm/generate_lm.py``.

For more information on how to build these resources from scratch, see the ``External scorer scripts`` section on `mozilla-voice-stt.readthedocs.io <https://mozilla-voice-stt.readthedocs.io/>`_.

