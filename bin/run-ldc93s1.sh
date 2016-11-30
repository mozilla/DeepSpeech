#!/bin/sh

set -xe

export ds_importer="ldc93s1"

export ds_train_batch_size=1
export ds_dev_batch_size=1
export ds_test_batch_size=1

export ds_training_iters=50

if [ ! -f DeepSpeech.ipynb ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u
