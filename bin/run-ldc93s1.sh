#!/bin/sh

set -xe

export ds_importer="ldc93s1"

export ds_train_batch_size=1
export ds_dev_batch_size=1
export ds_test_batch_size=1
export ds_n_hidden=494

export ds_epochs=50

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py
