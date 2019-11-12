#!/bin/sh

set -xe

ldc93s1_dir="./data/ldc93s1-tc"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --n_hidden 100 \
  --checkpoint_dir '/tmp/ckpt' \
  --export_dir '/tmp/train_tflite' \
  --lm_binary_path 'data/smoke_test/vocab.pruned.lm' \
  --lm_trie_path 'data/smoke_test/vocab.trie' \
  --export_tflite

mkdir /tmp/train_tflite/en-us

python -u DeepSpeech.py --noshow_progressbar \
  --n_hidden 100 \
  --checkpoint_dir '/tmp/ckpt' \
  --export_dir '/tmp/train_tflite/en-us' \
  --lm_binary_path 'data/smoke_test/vocab.pruned.lm' \
  --lm_trie_path 'data/smoke_test/vocab.trie' \
  --export_language 'Fake English (fk-FK)' \
  --export_zip
