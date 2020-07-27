#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_PROD_MODEL}
model_name=$(basename "${model_source}")

model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP}
model_name_mmap=$(basename "${model_source_mmap}")

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

check_versions

run_prod_inference_tests "${bitrate}"
