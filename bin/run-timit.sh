#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/TIMIT/timit_train.csv" ]; then
    echo "Trying to preprocess TIMIT data, saving in ./data/TIMIT"
    python -u bin/import_timit.py ./data/
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/timit"))')
fi

python -u DeepSpeech.py \
  --train_files data/TIMIT/timit_train.csv \
  --dev_files data/TIMIT/timit_test.csv \
  --test_files data/TIMIT/timit_test.csv \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --n_hidden 494 \
  --epoch 50 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
