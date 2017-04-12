#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

XDG_DATA_HOME=${XDG_DATA_HOME:="$HOME/.local/share"}

python -u DeepSpeech.py \
  --importer librivox \
  --train_batch_size 12 \
  --dev_batch_size 12 \
  --test_batch_size 12 \
  --learning_rate 0.0001 \
  --epoch 15 \
  --display_step 5 \
  --validation_step 5 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --checkpoint_dir "$XDG_DATA_HOME/deepspeech/librivox" \
  "$@"
