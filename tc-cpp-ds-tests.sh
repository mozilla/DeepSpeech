#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

aot_model=$1

download_material "${TASKCLUSTER_TMP_DIR}/ds" "${aot_model}"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

run_all_inference_tests
