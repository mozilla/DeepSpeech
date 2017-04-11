#!/bin/sh
set -xe
# ds_dataroot should have been set by cronjob
if [ ! -d "${ds_dataroot}" ]; then
    echo "No ${ds_dataroot} exist, aborting."
    exit 1
fi;

# Set temp directory to something where we can dump a lot.
TMP="${ds_dataroot}/tmp/"
mkdir -p "${TMP}" || true
export TMP

python -u DeepSpeech.py \
  --dataset_path="${ds_dataroot}/LIUM/" \
  --importer="ted" \
  --train_batch_size 16 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --epoch=50 \
  --learning_rate 0.0001 \
  --display_step 10 \
  --validation_step 10 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --checkpoint_step 1 \
  --export_dir "${ds_dataroot}/exports/`git rev-parse --short HEAD`" \
  "$@"

ln -sf $ds_export_dir $ds_export_dir/../latest
