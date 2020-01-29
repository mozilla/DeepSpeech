#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

check_deepspeech_version

run_basic_inference_tests
