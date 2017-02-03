#!/bin/sh

set -xe

export ds_importer="LDC97S62"

export ds_train_batch_size=48
export ds_dev_batch_size=32
export ds_test_batch_size=32

export ds_epochs=150
export ds_display_step=10
export ds_checkpoint_step=1

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py
