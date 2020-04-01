The LM binary was generated from the LibriSpeech normalized LM training text, available `here <http://www.openslr.org/11>`_.
It is created with `KenLM <https://github.com/kpu/kenlm>`_.

You can download the librispeech corpus with the following commands:

.. code-block:: bash

    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -O librispeech.txt.gz
    gunzip librispeech.txt.gz


Then use the `generate_lm.py` script to generate `lm.binary` and `vocab-500000.txt`.

.. code-block:: bash

    python3 data/lm/generate_lm.py --input_txt path/to/vocab_sentences.txt --output_dir path/lm/


Afterwards you can generate the scorer package with the above vocab-500000.txt and lm.binary files

.. code-block:: bash

    python generate_package.py --alphabet ../alphabet.txt --lm lm.binary --vocab vocab-500000.txt --package kenlm.scorer --default_alpha 0.75 --default_beta 1.85
