Language-Specific Data
======================

This directory contains language-specific data files. Most importantly, you will find here:

1. A list of unique characters for the target language (e.g. English) in ``data/alphabet.txt``. After installing the training code, you can check ``python -m deepspeech_training.util.check_characters --help`` for a tool that creates an alphabet file from a list of training CSV files.

2. A scorer package (``data/lm/kenlm.scorer``) generated with ``generate_scorer_package`` (``native_client/generate_scorer_package.cpp``). The scorer package includes a binary n-gram language model generated with ``data/lm/generate_lm.py``.

For more information on how to build these resources from scratch, see the ``External scorer scripts`` section on `deepspeech.readthedocs.io <https://deepspeech.readthedocs.io/>`_.

