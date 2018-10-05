#!/bin/bash

data="${DATA_ROOT}/shared/data"
fis="${data}/LDC/fisher"
swb="${data}/LDC/LDC97S62/swb"
lbs="${data}/OpenSLR/LibriSpeech/librivox"

target_dir="${DATA_ROOT}/groups/ds/cached"
mkdir -p "${target_dir}"

vocoto \
    add "${data}/UPF/freesound-cc0/ds.csv" \
    stash noise \
    add "${lbs}-train-clean-100.csv" \
    add "${lbs}-train-clean-360.csv" \
    add "${lbs}-train-other-500.csv" \
    add "${fis}-train.csv" \
    add "${swb}-train.csv" \
    shuffle \
    set original \
    hdf5 data/alphabet.txt "${target_dir}/ds_clean_train.hdf5" \
    shuffle \
    set crosstalk \
    \
    \
    shuffle \
    stash remaining \
    slice remaining 70 \
        augment noise -gain -5 \
    push result \
    clear \
    slice remaining 40 \
        augment crosstalk -times 10 -gain -10 \
    push result \
    clear \
    slice remaining 50 \
        distcompression 4 \
    push result \
    clear \
    add remaining \
    drop remaining \
        distrate 8000 \
    push result \
    clear \
    add result \
    hdf5 data/alphabet.txt "${target_dir}/ds_noise1_train.hdf5" \
    \
    \
    clear \
    drop result \
    add original \
    shuffle \
    stash remaining \
    slice remaining 70 \
        augment noise -times 2 \
    push result \
    clear \
    slice remaining 40 \
        augment crosstalk -times 5 -gain -5 \
    push result \
    clear \
    slice remaining 50 \
        distcompression 2 \
    push result \
    clear \
    add remaining \
    drop remaining \
        distrate 4000 \
    push result \
    clear \
    add result \
    hdf5 data/alphabet.txt "${target_dir}/ds_noise2_train.hdf5"

vocoto add "${lbs}-dev-clean.csv" hdf5 data/alphabet.txt "${target_dir}/ds_dev.hdf5"

vocoto add "${lbs}-test-clean.csv" hdf5 data/alphabet.txt "${target_dir}/ds_test.hdf5"
