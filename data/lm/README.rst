
lm.binary was generated from the LibriSpeech normalized LM training text, available `here <http://www.openslr.org/11>`_\ , using the `generate_lm.py` script (will generate lm.binary in the folder it is run from). KenLM's built binaries must be in your PATH (lmplz, build_binary, filter).

The trie was then generated from the vocabulary of the language model:

.. code-block:: bash

   ./generate_trie ../data/alphabet.txt lm.binary trie
