#!/bin/sh

set -xe

export ds_importer="fisher"

export ds_train_batch_size=32
export ds_dev_batch_size=32
export ds_test_batch_size=32

export ds_epochs=20
export ds_display_step=1
export ds_validation_step=5
export ds_checkpoint_step=5

if [ ! -f DeepSpeech.ipynb ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python -u
