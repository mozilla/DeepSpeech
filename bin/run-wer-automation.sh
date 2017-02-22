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

ds_dataset_path="${ds_dataroot}/LIUM/"
export ds_dataset_path

ds_importer="ted"
export ds_importer

ds_train_batch_size=16
export ds_train_batch_size

ds_dev_batch_size=8
export ds_dev_batch_size

ds_test_batch_size=8
export ds_test_batch_size

ds_epochs=50
export ds_epochs

ds_learning_rate=0.0001
export ds_learning_rate

ds_display_step=10
export ds_display_step

ds_validation_step=10
export ds_validation_step

ds_dropout_rate=0.30
export ds_dropout_rate

ds_default_stddev=0.046875
export ds_default_stddev

ds_checkpoint_step=1
export ds_checkpoint_step

ds_export_dir="${ds_dataroot}/exports/`git rev-parse --short HEAD`"
export ds_export_dir

python -u DeepSpeech.py

ln -sf $ds_export_dir $ds_export_dir/../latest
