#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

nodever=$1

if [ -z "${nodever}" ]; then
    echo "No node version given, aborting."
    exit 1
fi;

download_data

node --version
npm --version

NODE_ROOT="${DS_ROOT_TASK}/ds-test/"
export NODE_PATH="${NODE_ROOT}/node_modules/"
export PATH="${NODE_ROOT}:${NODE_PATH}/.bin/:$PATH"

npm install --prefix ${NODE_ROOT} ${DEEPSPEECH_NODEJS}/deepspeech-${DS_VERSION}.tgz

run_all_inference_tests
