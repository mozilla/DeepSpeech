#!/bin/bash

set -xe

# How to generate / update valgrind suppression lists:
# https://wiki.wxwidgets.org/Valgrind_Suppression_File_Howto#How_to_make_a_suppression_file
#
# $ valgrind --leak-check=full --show-reachable=yes --error-limit=no --gen-suppressions=all --log-file=minimalraw.log ./minimal
# $ cat ./minimalraw.log | ./parse_valgrind_suppressions.sh > minimal.supp

VALGRIND_CMD=${VALGRIND_CMD:-"valgrind \
    --error-exitcode=4242 \
    --errors-for-leak-kinds=all \
    --leak-check=full \
    --leak-resolution=high \
    --show-reachable=yes \
    --track-origins=yes \
    --gen-suppressions=all \
    --suppressions=${DS_DSDIR}/ds_generic.supp \
    --suppressions=${DS_DSDIR}/ds_lib.supp \
    --suppressions=${DS_DSDIR}/ds_sox.supp \
    --suppressions=${DS_DSDIR}/ds_openfst.supp \
    --suppressions=${DS_DSDIR}/tensorflow_full_runtime.supp \
    --suppressions=${DS_DSDIR}/tensorflow_tflite_runtime.supp \
"}

run_valgrind_basic()
{
    ${VALGRIND_CMD} --log-file=${TASKCLUSTER_ARTIFACTS}/valgrind_basic.log \
        deepspeech \
            --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
            --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
            --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} \
            -t
}

run_valgrind_stream()
{
    ${VALGRIND_CMD} --log-file=${TASKCLUSTER_ARTIFACTS}/valgrind_stream.log \
        deepspeech \
            --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
            --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
            --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} \
            --stream 320 \
            -t
}

run_valgrind_extended()
{
    ${VALGRIND_CMD} --log-file=${TASKCLUSTER_ARTIFACTS}/valgrind_extended.log \
        deepspeech \
            --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
            --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
            --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} \
            --extended \
            -t
}

run_valgrind_extended_stream()
{
    ${VALGRIND_CMD} --log-file=${TASKCLUSTER_ARTIFACTS}/valgrind_stream_extended.log \
        deepspeech \
            --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
            --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
            --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} \
            --extended_stream 320 \
            -t
}
