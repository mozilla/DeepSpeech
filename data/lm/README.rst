| Generate vocab-500000.txt and lm.binary files
|
| Optional Parameters:

* '--download_librispeech': Download the librispeech text corpus (will be downloaded to '--input_txt')
* '--kenlm_bins path/to/bins/':  Change the path of the kenlm binaries (defaults to directory in docker container)
* '--top_k 300000': Change the number of most frequent words
* '--arpa_order 3': Change order of k-grams in arpa-file generation
* '--max_arpa_memory 90%': Set maximum allowed memory usage in arpa-file generation


.. code-block:: bash

    python3 data/lm/generate_lm.py --input_txt path/to/vocab_sentences.txt --output_dir path/lm/


| Generate scorer package with the above vocab-500000.txt and lm.binary files

.. code-block:: bash

    python generate_package.py --alphabet ../alphabet.txt --lm lm.binary --vocab librispeech-vocab-500k.txt --default_alpha 0.75 --default_beta 1.85 --package kenlm.scorer
