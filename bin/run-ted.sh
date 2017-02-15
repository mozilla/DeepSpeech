#!/bin/sh

set -xe

export ds_importer="ted"

export ds_train_batch_size=16
export ds_dev_batch_size=8
export ds_test_batch_size=8

export ds_learning_rate=0.0001
export ds_validation_step=20

export ds_epochs=150
export ds_display_step=10
export ds_checkpoint_step=1

if [ ! -f DeepSpeech.ipynb ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u
