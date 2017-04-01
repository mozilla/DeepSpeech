#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${ds_dataroot}" ]; then
    ds_dataroot = "data"
fi;

python -u DeepSpeech.py \
  --train_files "$ds_dataroot/fisher-train.csv" \
  --dev_files "$ds_dataroot/fisher-dev.csv" \
  --test_files "$ds_dataroot/fisher-test.csv" \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --test_batch_size 32 \
  --learning_rate 0.0001 \
  --epoch=20 \
  --display_step 1 \
  --validation_step 5 \
  "$@"
