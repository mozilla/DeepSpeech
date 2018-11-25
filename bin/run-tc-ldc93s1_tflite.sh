#!/bin/sh

set -xe

ldc93s1_dir="./data/ldc93s1-tc"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

python -u DeepSpeech.py \
  --n_hidden 494 \
  --checkpoint_dir '/tmp/ckpt' \
  --export_dir '/tmp/train' \
  --lm_binary_path 'data/smoke_test/vocab.pruned.lm' \
  --lm_trie_path 'data/smoke_test/vocab.trie' \
  --notrain --notest \
  --export_tflite \
