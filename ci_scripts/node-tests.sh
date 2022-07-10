#!/bin/bash

set -xe

source $(dirname "$0")/all-vars.sh
source $(dirname "$0")/all-utils.sh
source $(dirname "$0")/asserts.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

download_data

node --version
npm --version

export_node_bin_path

check_runtime_nodejs

run_all_inference_tests

run_js_streaming_inference_tests

run_hotword_tests
