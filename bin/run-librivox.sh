#!/bin/sh

set -xe

ds_dataset_path="/data/OpenSLR/"
export ds_dataset_path

ds_importer="librivox"
export ds_importer

ds_train_batch_size=32
export ds_train_batch_size

ds_dev_batch_size=32
export ds_dev_batch_size

ds_test_batch_size=32
export ds_test_batch_size

ds_training_iters=15
export ds_training_iters

ds_validation_step=15
export ds_validation_step

if [ ! -f DeepSpeech.ipynb ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u
