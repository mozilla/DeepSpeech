
The LM binary was generated from the LibriSpeech normalized LM training text, available `here <http://www.openslr.org/11>`_\ , using the `generate_lm.py` script (will generate `lm.binary` and `librispeech-vocab-500k.txt` in the folder it is run from). `KenLM <https://github.com/kpu/kenlm>`_'s built binaries must be in your PATH (lmplz, build_binary, filter).

The scorer package was then built using the `generate_package.py` script:

.. code-block:: bash
    python generate_lm.py # this will create lm.binary and librispeech-vocab-500k.txt
    python generate_package.py --alphabet ../alphabet.txt --lm lm.binary --vocab librispeech-vocab-500k.txt --default_alpha 0.75 --default_beta 1.85 --package kenlm.scorer
