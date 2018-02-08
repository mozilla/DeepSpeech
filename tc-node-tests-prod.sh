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

node --version
npm --version
npm install ${DEEPSPEECH_NODEJS}/deepspeech-0.1.1.tgz

export PATH=$HOME/node_modules/.bin/:$PATH

run_prod_inference_tests
