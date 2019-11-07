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
NODE_CACHE="${DS_ROOT_TASK}/ds-test.cache/"
export NODE_PATH="${NODE_ROOT}/node_modules/"
export PATH="${NODE_ROOT}:${NODE_PATH}/.bin/:$PATH"

deepspeech_npm_url=$(get_dep_npm_pkg_url)
npm install --prefix ${NODE_ROOT} --cache ${NODE_CACHE} ${deepspeech_npm_url}

check_runtime_nodejs

run_all_inference_tests
