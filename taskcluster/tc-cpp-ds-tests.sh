#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

if [ "${OS}" = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$TESTS_BREW/lib/:$DYLD_LIBRARY_PATH
fi;

check_versions

run_all_inference_tests

run_multi_inference_tests

run_cpp_only_inference_tests
