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

pushd ${HOME}/DeepSpeech/ds/native_client/
    node --version
    npm --version
    if [ "${aot_model}" = "--aot" ]; then
        npm install ${DEEPSPEECH_AOT_ARTIFACTS_ROOT}/deepspeech-0.0.2.tgz
    else
        npm install ${DEEPSPEECH_ARTIFACTS_ROOT}/deepspeech-0.0.2.tgz
    fi

    npm install

    phrase_pbmodel_nolm=$(node client.js /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt)
    assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

    phrase_pbmodel_withlm=$(node client.js /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
    assert_correct_ldc93s1 "${phrase_pbmodel_withlm}"

    if [ "${aot_model}" = "--aot" ]; then
        phrase_somodel_nolm=$(node client.js "" /tmp/LDC93S1.wav /tmp/alphabet.txt)
        phrase_somodel_withlm=$(node client.js "" /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)

        assert_correct_ldc93s1_somodel "${phrase_somodel_nolm}" "${phrase_somodel_withlm}"
    fi
popd
