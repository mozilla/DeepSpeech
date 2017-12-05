#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

aot_model=$1

download_material "${TASKCLUSTER_TMP_DIR}/ds" "${aot_model}"

phrase_pbmodel_nolm=$(LD_LIBRARY_PATH=${TASKCLUSTER_TMP_DIR}/ds/:$LD_LIBRARY_PATH ${TASKCLUSTER_TMP_DIR}/ds/deepspeech ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt)
assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

phrase_pbmodel_withlm=$(LD_LIBRARY_PATH=${TASKCLUSTER_TMP_DIR}/ds/:$LD_LIBRARY_PATH ${TASKCLUSTER_TMP_DIR}/ds/deepspeech ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt ${TASKCLUSTER_TMP_DIR}/lm.binary ${TASKCLUSTER_TMP_DIR}/trie)
assert_correct_ldc93s1 "${phrase_pbmodel_withlm}"

if [ "${aot_model}" = "--aot" ]; then
    phrase_somodel_nolm=$(LD_LIBRARY_PATH=${TASKCLUSTER_TMP_DIR}/ds/:$LD_LIBRARY_PATH ${TASKCLUSTER_TMP_DIR}/ds/deepspeech "" ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt)
    phrase_somodel_withlm=$(LD_LIBRARY_PATH=${TASKCLUSTER_TMP_DIR}/ds/:$LD_LIBRARY_PATH ${TASKCLUSTER_TMP_DIR}/ds/deepspeech "" ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt ${TASKCLUSTER_TMP_DIR}/lm.binary ${TASKCLUSTER_TMP_DIR}/trie)

    assert_correct_ldc93s1_somodel "${phrase_somodel_nolm}" "${phrase_somodel_withlm}"
fi;
