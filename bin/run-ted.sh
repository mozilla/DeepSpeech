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
if [ ! -f "${ds_dataroot}/ted-train.csv" ]; then
    echo "Warning: It looks like you don't have the TED-LIUM corpus downloaded"\
         "and preprocessed. Make sure \$ds_dataroot points to the folder where"\
         "folder where the TED-LIUM data is located, and that you ran the" \
         "importer script at util/importers/ted.py before running this script."
fi;

python -u DeepSpeech.py \
  --train_files "$ds_dataroot/ted-train.csv" \
  --dev_files "$ds_dataroot/ted-dev.csv" \
  --test_files "$ds_dataroot/ted-test.csv" \
  --train_batch_size 16 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --epoch 10 \
  --display_step 10 \
  --validation_step 1 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --learning_rate 0.0001 \
  "$@"
