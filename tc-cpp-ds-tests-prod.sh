#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

model_source=${DEEPSPEECH_PROD_MODEL}
model_name=$(basename "${model_source}")

download_material "/tmp/ds"

phrase_pbmodel_withlm=$(LD_LIBRARY_PATH=/tmp/ds/:$LD_LIBRARY_PATH /tmp/ds/deepspeech /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}"
