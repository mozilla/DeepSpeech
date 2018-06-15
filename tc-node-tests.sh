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

NODE_ROOT="${DS_ROOT_TASK}/ds-test/"
export NODE_PATH="${NODE_ROOT}/node_modules/"
export PATH="${NODE_PATH}/.bin/:$PATH"

if [ "${aot_model}" = "--aot" ]; then
    npm install --prefix ${NODE_ROOT} ${DEEPSPEECH_AOT_ARTIFACTS_ROOT}/deepspeech-${DS_VERSION}.tgz
else
    npm install --prefix ${NODE_ROOT} ${DEEPSPEECH_NODEJS}/deepspeech-${DS_VERSION}.tgz
fi

run_all_inference_tests
