#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

nodever=$1
aot_model=$2

if [ -z "${nodever}" ]; then
    echo "No node version given, aborting."
    exit 1
fi;

download_data

node --version
npm --version
if [ "${aot_model}" = "--aot" ]; then
    npm install ${DEEPSPEECH_AOT_ARTIFACTS_ROOT}/deepspeech-0.1.0.tgz
else
    npm install ${DEEPSPEECH_NODEJS}/deepspeech-0.1.0.tgz
fi

export PATH=$HOME/node_modules/.bin/:$PATH

phrase_pbmodel_nolm=$(deepspeech /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt)
assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

phrase_pbmodel_withlm=$(deepspeech /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
assert_correct_ldc93s1 "${phrase_pbmodel_withlm}"

if [ "${aot_model}" = "--aot" ]; then
    phrase_somodel_nolm=$(deepspeech "" /tmp/LDC93S1.wav /tmp/alphabet.txt)
    phrase_somodel_withlm=$(deepspeech "" /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)

    assert_correct_ldc93s1_somodel "${phrase_somodel_nolm}" "${phrase_somodel_withlm}"
fi
