#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_PROD_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")
model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP//.pbmm/.tflite}
export DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_TFLITE_ROOT}
export DATA_TMP_DIR=${TASKCLUSTER_TMP_DIR}

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

check_versions

run_prodtflite_inference_tests "${bitrate}"
