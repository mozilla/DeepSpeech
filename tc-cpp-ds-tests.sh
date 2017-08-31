#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

download_material "/tmp/ds"

phrase=$(LD_LIBRARY_PATH=/tmp/ds/:$LD_LIBRARY_PATH /tmp/ds/deepspeech /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt)

assert_correct_ldc93s1 "${phrase}"
