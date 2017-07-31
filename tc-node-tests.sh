#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

nodever=$1

if [ -z "${nodever}" ]; then
    echo "No node version given, aborting."
    exit 1
fi;

download_material "/tmp/ds-lib"

phrase=""

pushd ${HOME}/DeepSpeech/ds/native_client/
    node --version
    npm --version
    npm install ${DEEPSPEECH_ARTIFACTS_ROOT}/deepspeech-0.0.1.tgz
    npm install
    phrase=$(LD_LIBRARY_PATH=/tmp/ds-lib/:$LD_LIBRARY_PATH node client.js /tmp/${model_name} /tmp/LDC93S1.wav)
popd

assert_correct_ldc93s1 "${phrase}"
