#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

nodever=$1

if [ -z "${nodever}" ]; then
    echo "No node version given, aborting."
    exit 1
fi;

model_source=${DEEPSPEECH_PROD_MODEL}
model_name=$(basename "${model_source}")

download_data

pushd ${HOME}/DeepSpeech/ds/native_client/
    node --version
    npm --version
    npm install ${DEEPSPEECH_ARTIFACTS_ROOT}/deepspeech-0.0.2.tgz
    npm install

    phrase_pbmodel_withlm=$(node client.js /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
    assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}"
popd
