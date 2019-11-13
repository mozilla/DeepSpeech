
lm.binary was generated from the LibriSpeech normalized LM training text, available `here <http://www.openslr.org/11>`_\ , following this recipe (Jupyter notebook code):

.. code-block:: python

   import gzip
   import io
   import os

   from urllib import request
   from collections import Counter

   # Grab corpus.
   url = 'http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz'
   data_upper = '/tmp/upper.txt.gz'
   request.urlretrieve(url, data_upper)

   # Convert to lowercase and count word occurences.
   counter = Counter()
   data_lower = '/tmp/lower.txt.gz'
   with io.TextIOWrapper(io.BufferedWriter(gzip.open(data_lower, 'w')), encoding='utf-8') as lower:
       with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
           for line in upper:
               line_lower = line.lower()
               counter.update(line_lower.split())
               lower.write(line_lower)

   # Create vocabulary file with top 500k words
   vocab_path = '/tmp/vocab-500k.txt'
   with open(vocab_path, 'w') as fout:
       fout.write('\n'.join(word for word, count in counter.most_common(500000)))

   # Build pruned LM.
   lm_path = '/tmp/lm.arpa'
   !lmplz --order 5 \
          --temp_prefix /tmp/ \
          --memory 50% \
          --text {data_lower} \
          --arpa {lm_path} \
          --prune 0 0 0 1

   # Filter LM using vocabulary.
   filtered_path = '/tmp/lm_filtered.arpa'
   !filter single model:{lm_path} {filtered_path} < {vocab_path}

   # Quantize and produce trie binary.
   binary_path = '/tmp/lm.binary'
   !build_binary -a 255 \
                 -q 8 \
                 trie \
                 {filtered_path} \
                 {binary_path} 
   os.remove(lm_path)

The trie was then generated from the vocabulary of the language model:

.. code-block:: bash

   ./generate_trie ../data/alphabet.txt /tmp/lm.binary /tmp/trie
