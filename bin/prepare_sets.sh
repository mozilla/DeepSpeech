#!/bin/bash

data="${SHARED_DIR}/data"
fis="${data}/LDC/fisher"
swb="${data}/LDC/LDC97S62/swb"
lbs="${data}/OpenSLR/LibriSpeech/librivox"

sets_dir="${ML_GROUP_DIR}/ds/training"
target_dir="${sets_dir}/augmented"
if [ -d "${target_dir}" ] ; then
    mv "${target_dir}" "${sets_dir}/augmented_$(date +"%Y%m%dT%H%M")"
fi
mkdir -p "${target_dir}"

git clone https://github.com/mozilla/voice-corpus-tool.git /tmp/vocoto
apt-get install -y libsndfile1 ffmpeg
pip3 install -r /tmp/vocoto/requirements.txt
vocoto () {
    python3 /tmp/vocoto/voice.py "$@"
}

process_set() {
    target_set=$1
    shift
    vocoto \
        add "${data}/UPF/freesound-cc0/ds.csv" \
        stash noise \
        "$@" \
        shuffle \
        set original \
        write "${target_dir}/ds_${target_set}_clean" \
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
        write "${target_dir}/ds_${target_set}_noise1" \
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
        write "${target_dir}/ds_${target_set}_noise2"

    vocoto add "${target_dir}/ds_${target_set}_clean.csv" \
           hdf5 data/alphabet.txt "${target_dir}/ds_${target_set}_clean.hdf5"

    vocoto add "${target_dir}/ds_${target_set}_noise1.csv" \
           hdf5 data/alphabet.txt "${target_dir}/ds_${target_set}_noise1.hdf5"

    vocoto add "${target_dir}/ds_${target_set}_noise2.csv" \
           hdf5 data/alphabet.txt "${target_dir}/ds_${target_set}_noise2.hdf5"
}

process_set train \
    add "${fis}-train.csv" \
    add "${swb}-train.csv" \
    add "${lbs}-train-clean-100.csv" \
    add "${lbs}-train-clean-360.csv" \
    add "${lbs}-train-other-500.csv"

process_set dev \
    add "${lbs}-dev-clean.csv" \
    add "${lbs}-dev-other.csv"

vocoto \
    add "${lbs}-test-clean.csv" \
    add "${lbs}-test-other.csv" \
    write "${target_dir}/ds_test"

vocoto add "${target_dir}/ds_test.csv" \
    hdf5 data/alphabet.txt "${target_dir}/ds_test.hdf5"
