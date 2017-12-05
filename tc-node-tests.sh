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
    npm install --prefix ${NODE_ROOT} ${DEEPSPEECH_AOT_ARTIFACTS_ROOT}/deepspeech-0.1.0.tgz
else
    npm install --prefix ${NODE_ROOT} ${DEEPSPEECH_NODEJS}/deepspeech-0.1.0.tgz
fi

phrase_pbmodel_nolm=$(deepspeech ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt)
assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

phrase_pbmodel_withlm=$(deepspeech ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt ${TASKCLUSTER_TMP_DIR}/lm.binary ${TASKCLUSTER_TMP_DIR}/trie)
assert_correct_ldc93s1 "${phrase_pbmodel_withlm}"

if [ "${aot_model}" = "--aot" ]; then
    phrase_somodel_nolm=$(deepspeech "" ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt)
    phrase_somodel_withlm=$(deepspeech "" ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt ${TASKCLUSTER_TMP_DIR}/lm.binary ${TASKCLUSTER_TMP_DIR}/trie)

    assert_correct_ldc93s1_somodel "${phrase_somodel_nolm}" "${phrase_somodel_withlm}"
fi
