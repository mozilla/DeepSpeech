.. _decoder-docs:

CTC beam search decoder
=======================

Introduction
^^^^^^^^^^^^

Mozilla Voice STT uses the `Connectionist Temporal Classification <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ loss function. For an excellent explanation of CTC and its usage, see this Distill article: `Sequence Modeling with CTC <https://distill.pub/2017/ctc/>`_. This document assumes the reader is familiar with the concepts described in that article, and describes Mozilla Voice STT specific behaviors that developers building systems with Mozilla Voice STT should know to avoid problems.

Note: Documentation for the tooling for creating custom scorer packages is available in :ref:`scorer-scripts`.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED",  "MAY", and "OPTIONAL" in this document are to be interpreted as described in `BCP 14 <https://tools.ietf.org/html/bcp14>`_ when, and only when, they appear in all capitals, as shown here.


External scorer
^^^^^^^^^^^^^^^

Mozilla Voice STT clients support OPTIONAL use of an external language model to improve the accuracy of the predicted transcripts. In the code, command line parameters, and documentation, this is referred to as a "scorer". The scorer is used to compute the likelihood (also called a score, hence the name "scorer") of sequences of words or characters in the output, to guide the decoder towards more likely results. This improves accuracy significantly.

The use of an external scorer is fully optional. When an external scorer is not specified, Mozilla Voice STT still uses a beam search decoding algorithm, but without any outside scoring.

Currently, the Mozilla Voice STT external scorer is implemented with `KenLM <https://kheafield.com/code/kenlm/>`_, plus some tooling to package the necessary files and metadata into a single ``.scorer`` package. The tooling lives in ``data/lm/``. The scripts included in ``data/lm/`` can be used and modified to build your own language model based on your particular use case or language. See :ref:`scorer-scripts` for more details on how to reproduce our scorer file as well as create your own.

The scripts are geared towards replicating the language model files we release as part of `Mozilla Voice STT model releases <https://github.com/mozilla/DeepSpeech/releases/latest>`_, but modifying them to use different datasets or language model construction parameters should be simple.


Decoding modes
^^^^^^^^^^^^^^

Mozilla Voice STT currently supports two modes of operation with significant differences at both training and decoding time. Note that Bytes output mode is experimental and has not been tested for languages other than Chinese Mandarin.


Default mode (alphabet based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default mode, which uses an alphabet file (specified with ``--alphabet_config_path`` at training and export time) to determine which labels (characters), and how many of them, to predict in the output layer. At decoding time, if using an external scorer, it MUST be word based and MUST be built using the same alphabet file used for training. Word based means the text corpus used to build the scorer should contain words separated by whitespace. For most western languages, this is the default and requires no special steps from the developer when creating the scorer.


Bytes output mode
^^^^^^^^^^^^^^^^^

**Note**: Currently, Bytes output mode makes assumptions that hold for Chinese Mandarin models but do not hold for other language targets, such as not predicting spaces.

In bytes output mode the model predicts UTF-8 bytes directly instead of letters from an alphabet file. This idea was proposed in the paper `Bytes Are All You Need <https://arxiv.org/abs/1811.09021>`_. This mode is enabled with the ``--utf8`` flag at training and export time. At training time, the alphabet file is not used. Instead, the model is forced to have 256 labels, with labels 0-254 corresponding to UTF-8 byte values 1-255, and label 255 is used for the CTC blank symbol. If using an external scorer at decoding time, it MUST be built according to the instructions that follow.

Bytes output mode can be useful for languages with very large alphabets, such as Mandarin written with Simplified Chinese characters. It may also be useful for building multi-language models, or as a base for transfer learning. Currently these cases are untested and unsupported. Note that bytes output mode makes assumptions that hold for Mandarin written with Simplified Chinese characters and may not hold for other languages.

UTF-8 scorers are character based (more specifically, Unicode codepoint based), but the way they are used is similar to a word based scorer where each "word" is a sequence of UTF-8 bytes representing a single Unicode codepoint. This means that the input text used to create UTF-8 scorers should contain space separated Unicode codepoints. For example, the following input text:

``早 上 好``

corresponds to the following three "words", or UTF-8 byte sequences:

``E6 97 A9``
``E4 B8 8A``
``E5 A5 BD``

At decoding time, the scorer is queried every time a Unicode codepoint is predicted, instead of when a space character is predicted. From the language modeling perspective, this is a character based model. From the implementation perspective, this is a word based model, because each character is composed of multiple labels.

**Acoustic models trained with ``--utf8`` MUST NOT be used with an alphabet based scorer. Conversely, acoustic models trained with an alphabet file MUST NOT be used with a UTF-8 scorer.**

UTF-8 scorers can be built by using an input corpus with space separated codepoints. If your corpus only contains single codepoints separated by spaces, ``generate_scorer_package`` should automatically enable bytes output mode, and it should print the message "Looks like a character based model."

If the message "Doesn't look like a character based model." is printed, you should double check your inputs to make sure it only contains single codepoints separated by spaces. Bytes output mode can be forced by specifying the ``--force_utf8`` flag when running ``generate_scorer_package``, but it is NOT RECOMMENDED.

See :ref:`scorer-scripts` for more details on using ``generate_scorer_package``.

Because KenLM uses spaces as a word separator, the resulting language model will not include space characters in it. If you wish to use bytes output mode but still model spaces, you need to replace spaces in the input corpus with a different character **before** converting it to space separated codepoints. For example:

.. code-block:: python

   input_text = 'The quick brown fox jumps over the lazy dog'
   spaces_replaced = input_text.replace(' ', '|')
   space_separated = ' '.join(spaces_replaced)
   print(space_separated)
   # T h e | q u i c k | b r o w n | f o x | j u m p s | o v e r | t h e | l a z y | d o g

The character, '|' in this case, will then have to be replaced with spaces as a post-processing step after decoding.


Implementation
^^^^^^^^^^^^^^

The decoder source code can be found in ``native_client/ctcdecode``. The decoder is included in the language bindings and clients. In addition, there is a separate Python module which includes just the decoder and is needed for evaluation. A pre-built version of this package is automatically downloaded and installed when installing the training code. If you want or need to manually build and install it from source, see the :github:`native_client README <native_client/README.rst#install-the-ctc-decoder-package>`.
