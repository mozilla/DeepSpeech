#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

COMPUTE_KEEP_DIR=~/data/compute
if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

python ./DeepSpeech.py  \
  --checkpoint_dir "$checkpoint_dir" \
  --decoder_library_path '/home/ubuntu/DeepSpeech/native_client_bin/libctc_decoder_with_kenlm.so'  \
  --fulltrace true \
  --coord_port 9999 \
  --notrain \
  --notest \
  --one-shot-infer /home/ubuntu/data/cv_corpus_v1/cv-valid-test/sample-002160.wav

