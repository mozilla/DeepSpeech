| Generate vocab-500000.txt and lm.binary files
| Add '--download_librispeech' to download the librispeech text corpus (will be downloaded to '--input_txt')
| Optional change the path of the kenlm binaries with '--kenlm_bins path/to/bins/'
| Optional change the number of most frequent words with  '--top_k 300000'


.. code-block:: bash

    python3 data/lm/generate_lm.py --input_txt path/to/vocab_sentences.txt --output_dir path/lm/


| Generate scorer package with the above vocab-500000.txt and lm.binary files

.. code-block:: bash

    python generate_package.py --alphabet ../alphabet.txt --lm lm.binary --vocab librispeech-vocab-500k.txt --default_alpha 0.75 --default_beta 1.85 --package kenlm.scorer
