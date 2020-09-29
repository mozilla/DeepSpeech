#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

set_ldc_sample_filename "16k"

download_material "${TASKCLUSTER_TMP_DIR}/ds"

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

run_valgrind_basic

run_valgrind_stream
