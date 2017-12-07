#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

model_source=${DEEPSPEECH_PROD_MODEL}
model_name=$(basename "${model_source}")

download_material "${TASKCLUSTER_TMP_DIR}/ds"

phrase_pbmodel_withlm=$(LD_LIBRARY_PATH=${TASKCLUSTER_TMP_DIR}/ds/:$LD_LIBRARY_PATH ${TASKCLUSTER_TMP_DIR}/ds/deepspeech ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt ${TASKCLUSTER_TMP_DIR}/lm.binary ${TASKCLUSTER_TMP_DIR}/trie)
assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}"
