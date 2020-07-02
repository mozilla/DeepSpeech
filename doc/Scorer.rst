.. _scorer-scripts:

External scorer scripts
=======================

DeepSpeech pre-trained models include an external scorer. This document explains how to reproduce our external scorer, as well as adapt the scripts to create your own.

The scorer is composed of two sub-components, a KenLM language model and a trie data structure containing all words in the vocabulary. In order to create the scorer package, first we must create a KenLM language model (using ``data/lm/generate_lm.py``, and then use ``generate_scorer_package`` to create the final package file including the trie data structure.

The ``generate_scorer_package`` binary is part of the native client package that is included with official releases. You can find the appropriate archive for your platform in the `GitHub release downloads <https://github.com/mozilla/DeepSpeech/releases/latest>`_. The native client package is named ``native_client.{arch}.{config}.{plat}.tar.xz``, where ``{arch}`` is the architecture the binary was built for, for example ``amd64`` or ``arm64``, ``config`` is the build configuration, which for building decoder packages does not matter, and ``{plat}`` is the platform the binary was built-for, for example ``linux`` or ``osx``. If you wanted to run the ``generate_scorer_package`` binary on a Linux desktop, you would download ``native_client.amd64.cpu.linux.tar.xz``.

Reproducing our external scorer
-------------------------------

Our KenLM language model was generated from the LibriSpeech normalized LM training text, available `here <http://www.openslr.org/11>`_.
It is created with `KenLM <https://github.com/kpu/kenlm>`_.

You can download the LibriSpeech corpus with the following command:

.. code-block:: bash

    cd data/lm
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz

Then use the ``generate_lm.py`` script to generate ``lm.binary`` and ``vocab-500000.txt``.

As input you can use a plain text (e.g. ``file.txt``) or gzipped (e.g. ``file.txt.gz``) text file with one sentence in each line.

If you are using a container created from ``Dockerfile.build``, you can use ``--kenlm_bins /DeepSpeech/native_client/kenlm/build/bin/``.
Else you have to build `KenLM <https://github.com/kpu/kenlm>`_ first and then pass the build directory to the script.

.. code-block:: bash

    cd data/lm
    python3 generate_lm.py --input_txt librispeech-lm-norm.txt.gz --output_dir . \
      --top_k 500000 --kenlm_bins path/to/kenlm/build/bin/ \
      --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
      --binary_a_bits 255 --binary_q_bits 8 --binary_type trie


Afterwards you can use ``generate_scorer_package`` to generate the scorer package using the ``lm.binary`` and ``vocab-500000.txt`` files:

.. code-block:: bash

    cd data/lm
    # Download and extract appropriate native_client package:
    curl -LO http://github.com/mozilla/DeepSpeech/releases/...
    tar xvf native_client.*.tar.xz
    ./generate_scorer_package --alphabet ../alphabet.txt --lm lm.binary --vocab vocab-500000.txt \
      --package kenlm.scorer --default_alpha 0.931289039105002 --default_beta 1.1834137581510284

Building your own scorer
------------------------

Building your own scorer can be useful if you're using models in a narrow usage context, with a more limited vocabulary, for example. Building a scorer requires text data matching your intended use case, which must be formatted in a text file with one sentence per line.

The LibriSpeech LM training text used by our scorer is around 4GB uncompressed, which should give an idea of the size of a corpus needed for a reasonable language model for general speech recognition. For more constrained use cases with smaller vocabularies, you don't need as much data, but you should still try to gather as much as you can.

With a text corpus in hand, you can then re-use ``generate_lm.py`` and ``generate_scorer_package`` to create your own scorer that is compatible with DeepSpeech clients and language bindings. Before building the language model, you must first familiarize yourself with the `KenLM toolkit <https://kheafield.com/code/kenlm/>`_. Most of the options exposed by the ``generate_lm.py`` script are simply forwarded to KenLM options of the same name, so you must read the KenLM documentation in order to fully understand their behavior.

After using ``generate_lm.py`` to create a KenLM language model binary file, you can use ``generate_scorer_package`` to create a scorer package as described in the previous section. Note that we have a :github:`lm_optimizer.py script <lm_optimizer.py>` which can be used to find good default values for alpha and beta. To use it, you must first generate a package with any value set for default alpha and beta flags. For this step, it doesn't matter what values you use, as they'll be overridden by ``lm_optimizer.py`` later. Then, use ``lm_optimizer.py`` with this scorer file to find good alpha and beta values. Finally, use ``generate_scorer_package`` again, this time with the new values.
