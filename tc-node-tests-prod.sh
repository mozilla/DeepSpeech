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
npm install ${DEEPSPEECH_NODEJS}/deepspeech-0.1.0.tgz

export PATH=$HOME/node_modules/.bin/:$PATH

phrase_pbmodel_withlm=$(deepspeech ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt ${TASKCLUSTER_TMP_DIR}/lm.binary ${TASKCLUSTER_TMP_DIR}/trie)
assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}"
