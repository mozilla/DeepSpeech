#!/bin/sh

set -xe

ds_dataset_path="./data/ldc93s1/"
export ds_dataset_path

ds_importer="ldc93s1"
export ds_importer

ds_train_batch_size=4
export ds_train_batch_size

ds_dev_batch_size=4
export ds_dev_batch_size

ds_test_batch_size=4
export ds_test_batch_size

ds_training_iters=5
export ds_training_iters

if [ ! -f DeepSpeech.ipynb ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u
