#!/bin/sh

set -xe

ds_dataset_path="/data/LIUM/"
export ds_dataset_path

ds_importer="ted"
export ds_importer

ds_batch_size=64
export ds_batch_size

ds_training_iters=30
export ds_training_iters

ds_validation_step=15
export ds_validation_step

ds_learning_rate=0.01
export ds_learning_rate

ds_display_step=10
export ds_display_step

ds_export_dir="/data/exports/`git rev-parse --short HEAD`"
export ds_export_dir

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u

ln -sf $ds_export_dir $ds_export_dir/../latest
