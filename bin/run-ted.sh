#!/bin/sh

set -xe

export ds_importer="ted"

export ds_train_batch_size=16
export ds_dev_batch_size=8
export ds_test_batch_size=8

export ds_learning_rate=0.0001

export ds_epochs=50
export ds_validation_step=10
export ds_display_step=10

export ds_dropout_rate=0.30
export ds_checkpoint_step=1

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py
