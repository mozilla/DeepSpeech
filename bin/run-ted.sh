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
if [ ! -f "${COMPUTE_DATA_DIR}/ted-train.csv" ]; then
    echo "Warning: It looks like you don't have the TED-LIUM corpus downloaded"\
         "and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the folder where"\
         "folder where the TED-LIUM data is located, and that you ran the" \
         "importer script at bin/import_ted.py before running this script."
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ted"))')
fi

python -u DeepSpeech.py \
  --train_files "$COMPUTE_DATA_DIR/ted-train.csv" \
  --dev_files "$COMPUTE_DATA_DIR/ted-dev.csv" \
  --test_files "$COMPUTE_DATA_DIR/ted-test.csv" \
  --train_batch_size 16 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --epoch 10 \
  --display_step 10 \
  --validation_step 1 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --learning_rate 0.0001 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
