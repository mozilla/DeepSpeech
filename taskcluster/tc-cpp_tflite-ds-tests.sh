#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")
export DATA_TMP_DIR=${TASKCLUSTER_TMP_DIR}

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

check_versions

run_all_inference_tests

run_multi_inference_tests

run_cpp_only_inference_tests
