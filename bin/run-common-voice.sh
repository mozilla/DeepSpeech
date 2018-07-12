#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "~/data/cv-valid-train.csv" ]; then
    echo "You need to download and preprocess common voice data into ~/data"
    #python -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

COMPUTE_KEEP_DIR=~/data/compute
if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

python ./DeepSpeech.py  \
  --train_files '~/data/cv-valid-train.csv' \
  --dev_files '~/data/cv-valid-dev.csv' \
  --test_files '~/data/cv-valid-test.csv' \
  --train_batch_size 12 \
  --dev_batch_size 12 \
  --test_batch_size 12 \
  --learning_rate 0.0001 \
  --epoch 15 \
  --display_step 5 \
  --validation_step 5 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --checkpoint_dir "$checkpoint_dir" \
  --decoder_library_path '/home/ubuntu/DeepSpeech/native_client_bin/libctc_decoder_with_kenlm.so'  \
  --summary_dir '/home/ubuntu/tboard'  \
  --summary_secs 60 \
  --fulltrace true
  "$@"

