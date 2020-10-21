#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

nodever=$1

if [ -z "${nodever}" ]; then
    echo "No node version given, aborting."
    exit 1
fi;

bitrate=$2
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")

download_data

node --version
npm --version

NODE_ROOT="${DS_ROOT_TASK}/ds-test/"
NODE_CACHE="${DS_ROOT_TASK}/ds-test.cache/"
export NODE_PATH="${NODE_ROOT}/node_modules/"
export PATH="${NODE_ROOT}:${NODE_PATH}/.bin/:$PATH"

# make sure that NODE_ROOT really exists
mkdir -p ${NODE_ROOT}

deepspeech_npm_url=$(get_dep_npm_pkg_url)
npm install --prefix ${NODE_ROOT} --cache ${NODE_CACHE} ${deepspeech_npm_url}

check_runtime_nodejs

run_all_inference_tests

run_js_streaming_inference_tests

run_hotword_tests
