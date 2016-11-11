#!/bin/sh

set -xe

ds_dataset_path="/data/LIUM/"
export ds_dataset_path

ds_importer="ted"
export ds_importer

ds_batch_size=32
export ds_batch_size

ds_training_iters=15
export ds_training_iters

ds_validation_step=15
export ds_validation_step

ds_export_dir="/data/exports/`git rev-parse --short HEAD`"
export ds_export_dir

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python

ln -sf $ds_export_dir $ds_export_dir/../latest
