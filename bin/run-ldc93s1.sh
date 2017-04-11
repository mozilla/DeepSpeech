#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

XDG_DATA_HOME=${XDG_DATA_HOME:="$HOME/DeepSpeech"}

python -u DeepSpeech.py \
  --importer ldc93s1 \
  --train_batch_size 1 \
  --dev_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 494 \
  --epoch 50 \
  --checkpoint_dir "$XDG_DATA_HOME/$(basename ${0%.*})" \
  "$@"
