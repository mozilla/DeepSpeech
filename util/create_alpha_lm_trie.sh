#!/bin/bash

LANG=$1
TEXT=$2

echo "$0: Looking for CSV transcripts at cv_${LANG}_valid_{train/dev/test}.csv"
echo "$0: Looking for text training corpus at ${TEXT}"

# kenlm Dependencies
apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev libeigen3-dev

# Install Kenlm #

wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j `nproc`
cd ../..

#################
### CREATE LM ###
#################

# Make alphabet.txt #

python3 util/check_characters.py \
        -csv "cv_${LANG}_valid_train.csv","cv_${LANG}_valid_train.csv","cv_${LANG}_valid_train.csv" \
        -alpha \
    | data/alphabet.txt

# Make lm.arpa #

kenlm/build/bin/lmplz \
    --order 2 \
    --text ${TEXT} \
    --arpa /tmp/lm.arpa

# Make lm.binary #

kenlm/build/bin/build_binary \
    -a 255 \
    -q 8 trie \
    /tmp/lm.arpa \
    data/lm/lm.binary

# Make trie #

native_client/generate_trie \
    data/alphabet.txt \
    data/lm/lm.binary \
    data/lm/trie

rm /tmp/lm.arpa


