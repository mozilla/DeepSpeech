# Language-Specific Data

This directory contains language-specific data files. Most importantly, you will find here:

1. A list of unique characters for the target language (e.g. English) in `data/alphabet.txt`
2. A binary n-gram language model compiled by `kenlm` in `data/lm/lm.binary`
3. A trie model compiled by `generate_trie.cpp` in `data/lm/trie`

For more information on how to create these resources, see `data/lm/README.md`
