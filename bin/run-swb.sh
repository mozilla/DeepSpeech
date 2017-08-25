#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="data"
fi;

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/swb-train.csv" ]; then
    echo "Warning: It looks like you don't have the Switchboard corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Switchboard data is located, and that you ran the" \
         "importer script at bin/import_swb.py before running this script."
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/swb"))')
fi

python -u DeepSpeech.py \
  --train_files "${COMPUTE_DATA_DIR}/LDC/LDC97S62/swb-train.csv" \
  --dev_files "${COMPUTE_DATA_DIR}/LDC/LDC97S62/swb-dev.csv" \
  --test_files "${COMPUTE_DATA_DIR}/LDC/LDC97S62/swb-test.csv" \
  --train_batch_size 13 \
  --dev_batch_size 13 \
  --test_batch_size 13 \
  --epoch 15 \
  --learning_rate 0.0001 \
  --display_step 0 \
  --validation_step 1 \
  --dropout_rate 0.15 \
  --default_stddev 0.046875 \
  --checkpoint_step 1 \
  --checkpoint_dir "${COMPUTE_KEEP_DIR}" \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)"\
  "$@"
