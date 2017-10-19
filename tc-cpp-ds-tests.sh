#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

download_material "/tmp/ds"

phrase_pbmodel_withlm=$(LD_LIBRARY_PATH=/tmp/ds/:$LD_LIBRARY_PATH /tmp/ds/deepspeech /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
assert_correct_ldc93s1 "${phrase_pbmodel_withlm}"

phrase_pbmodel_nolm=$(LD_LIBRARY_PATH=/tmp/ds/:$LD_LIBRARY_PATH /tmp/ds/deepspeech /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt)
assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

phrase_somodel_withlm=$(LD_LIBRARY_PATH=/tmp/ds/:$LD_LIBRARY_PATH /tmp/ds/deepspeech "" /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
assert_correct_ldc93s1 "${phrase_somodel_withlm}"

phrase_somodel_nolm=$(LD_LIBRARY_PATH=/tmp/ds/:$LD_LIBRARY_PATH /tmp/ds/deepspeech "" /tmp/LDC93S1.wav /tmp/alphabet.txt)
assert_correct_ldc93s1 "${phrase_somodel_nolm}"
