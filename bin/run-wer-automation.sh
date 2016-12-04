#!/bin/sh

set -xe

ds_dataset_path="/data/LIUM/"
export ds_dataset_path

ds_importer="ted"
export ds_importer

ds_train_batch_size=48
export ds_train_batch_size

ds_dev_batch_size=32
export ds_dev_batch_size

ds_test_batch_size=32
export ds_test_batch_size

ds_epochs=500
export ds_epochs

ds_learning_rate=0.0001
export ds_learning_rate

ds_display_step=10
export ds_display_step

ds_validation_step=20
export ds_validation_step

ds_dropout_rate=0.05
export ds_dropout_rate

ds_default_stddev=0.1
export ds_default_stddev

ds_use_warpctc=1
export ds_use_warpctc

ds_checkpoint_step=1
export ds_checkpoint_step

ds_export_dir="/data/exports/`git rev-parse --short HEAD`"
export ds_export_dir

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u

ln -sf $ds_export_dir $ds_export_dir/../latest
