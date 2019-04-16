#!/bin/sh

num_bytes=$1
NEW_TMP=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)


mkdir /tmp/$NEW_TMP

python -u DeepSpeech.py --noshow_progressbar \
  --train_files data/ldc93s1/${num_bytes}-byte/ldc93s1.csv \
  --test_files data/ldc93s1/${num_bytes}-byte/ldc93s1.csv \
  --n_layers 1 \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 1000 \
  --checkpoint_dir /tmp/$NEW_TMP \
  "$@"

rm -rf /tmp/$NEW_TMP
