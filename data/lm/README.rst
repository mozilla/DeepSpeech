Download librispeech corpus

.. code-block:: bash

    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -O librispeech.txt.gz
    gunzip librispeech.txt.gz

|
| Generate vocab-500000.txt and lm.binary files
| Optional Parameters:

* '--kenlm_bins path/to/bins/':  Change the path of the kenlm binaries (defaults to directory in docker container)
* '--top_k 500000': Change the number of most frequent words
* '--arpa_order 5': Change order of k-grams in arpa-file generation
* '--max_arpa_memory 75%': Set maximum allowed memory usage for arpa-file generation


.. code-block:: bash

    python3 data/lm/generate_lm.py --input_txt path/to/vocab_sentences.txt --output_dir path/lm/

|
| Generate scorer package with the above vocab-500000.txt and lm.binary files
| Optional Parameters:

* '--default_alpha 0.75'
* '--default_beta 1.85'
* '--force_utf8 ""': See `link <https://github.com/mozilla/DeepSpeech/blob/master/doc/Decoder.rst#utf-8-mode>`_ for explanation

.. code-block:: bash

    python generate_package.py --alphabet ../alphabet.txt --lm lm.binary --vocab librispeech-vocab-500k.txt --package kenlm.scorer
