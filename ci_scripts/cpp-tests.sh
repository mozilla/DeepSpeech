#!/bin/bash

set -xe

source $(dirname "$0")/all-vars.sh
source $(dirname "$0")/all-utils.sh
source $(dirname "$0")/asserts.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

download_data

export PATH=${CI_TMP_DIR}/ds/:$PATH

check_versions

run_all_inference_tests

run_multi_inference_tests

run_cpp_only_inference_tests

run_hotword_tests
