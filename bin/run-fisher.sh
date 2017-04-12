#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/fisher"))')

python -u DeepSpeech.py \
  --importer fisher \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --test_batch_size 32 \
  --learning_rate 0.0001 \
  --epoch=20 \
  --display_step 1 \
  --validation_step 5 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
