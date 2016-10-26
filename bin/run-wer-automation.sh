#!/bin/sh

set -xe

ds_dataset_path="/data/ted/"
export ds_dataset_path

ds_importer="ted_lium"
export ds_importer

ds_batch_size=64
export ds_batch_size

ds_training_iters=15
export ds_training_iters

jupyter-nbconvert --to script DeepSpeech.ipynb --stdout | python
