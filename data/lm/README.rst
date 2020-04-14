| The LM binary was generated from the LibriSpeech normalized LM training text, available `here <http://www.openslr.org/11>`_.
| It is created with `KenLM <https://github.com/kpu/kenlm>`_.
|

You can download the LibriSpeech corpus with the following commands:

.. code-block:: bash

    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz


| Then use the ``generate_lm.py`` script to generate ``lm.binary`` and ``vocab-500000.txt``.
| As input you can use a plain text (e.g. ``file.txt``) or gzipped (e.g. ``file.txt.gz``) text file with one sentence in each line.
|
| If you are using a container created from the Dockerfile, you can use ``--kenlm_bins /DeepSpeech/native_client/kenlm/build/bin/``.
 Else you have to build `KenLM <https://github.com/kpu/kenlm>`_ first and then pass the build directory to the script.

.. code-block:: bash

    python3 generate_lm.py --input_txt librispeech-lm-norm.txt.gz --output_dir . \
      --top_k 500000 --kenlm_bins path/to/kenlm/build/bin/ \
      --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
      --binary_a_bits 255 --binary_q_bits 8 --binary_type trie


Afterwards you can use ``generate_package.py`` to generate the scorer package using the ``lm.binary`` and ``vocab-500000.txt`` files:

.. code-block:: bash

    python3 generate_package.py --alphabet ../alphabet.txt --lm lm.binary --vocab vocab-500000.txt \
      --package kenlm.scorer --default_alpha 0.75 --default_beta 1.85
