#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${ds_dataroot}" ]; then
    ds_dataroot="data"
fi;

# Warn if we can't find the train files
if [ ! -f "${ds_dataroot}/fisher-train.csv" ]; then
    echo "Warning: It looks like you don't have the Fisher corpus"            \
         "downloaded and preprocessed. Make sure \$ds_dataroot points to the" \
         "folder where the Fisher data is located, and that you ran the"      \
         "importer script at bin/import_fisher.py before running this script."
fi;

checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/fisher"))')

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
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
