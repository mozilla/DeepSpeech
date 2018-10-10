#!/bin/bash

data="${DATA_ROOT}/shared/data"
fis="${data}/LDC/fisher"
swb="${data}/LDC/LDC97S62/swb"
lbs="${data}/OpenSLR/LibriSpeech/librivox"
cv="${data}/mozilla/CommonVoice/v2.0-alpha1.0/en/cv_en_valid_"

target_dir="$1"

process_set() {
    target_set=$1
    shift
    vocoto \
        add "${data}/UPF/freesound-cc0/ds.csv" \
        stash noise \
        "$@" \
        shuffle \
        set original \
        hdf5 data/alphabet.txt "${target_dir}/ds_${target_set}_clean.hdf5" \
        shuffle \
        set crosstalk \
        \
        shuffle \
        stash remaining \
        slice remaining 80 \
            augment noise -gain -5 \
        push result \
        clear \
        slice remaining 80 \
            augment crosstalk -times 10 -gain -10 \
        push result \
        clear \
        slice remaining 20 \
            compr 4 \
        push result \
        clear \
        add remaining \
        drop remaining \
            rate 8000 \
            rate 16000 \
        push result \
        clear \
        add result \
        hdf5 data/alphabet.txt "${target_dir}/ds_${target_set}_noise1.hdf5" \
        \
        clear \
        drop result \
        add original \
        shuffle \
        stash remaining \
        slice remaining 80 \
            augment noise -times 2 \
        push result \
        clear \
        slice remaining 80 \
            augment crosstalk -times 5 -gain -5 \
        push result \
        clear \
        slice remaining 20 \
            compr 2 \
        push result \
        clear \
        add remaining \
        drop remaining \
            rate 4000 \
            rate 16000 \
        push result \
        clear \
        add result \
        hdf5 data/alphabet.txt "${target_dir}/ds_${target_set}_noise2.hdf5"
}

process_set train \
    add "${fis}-train.csv" \
    add "${swb}-train.csv" \
    add "${cv}train.csv" \
    add "${lbs}-train-clean-100.csv" \
    add "${lbs}-train-clean-360.csv" \
    add "${lbs}-train-other-500.csv"

process_set dev \
    add "${cv}dev.csv" \
    add "${lbs}-dev-clean.csv" \
    add "${lbs}-dev-other.csv"

vocoto \
    add "${cv}test.csv" \
    add "${lbs}-test-clean.csv" \
    add "${lbs}-test-other.csv" \
    hdf5 data/alphabet.txt "${target_dir}/ds_test.hdf5"
