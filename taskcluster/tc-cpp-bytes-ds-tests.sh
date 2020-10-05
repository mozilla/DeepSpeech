#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

# Bytes output mode with LDC93S1 takes too long to converge so we simply test
# that loading the model won't crash
check_versions
