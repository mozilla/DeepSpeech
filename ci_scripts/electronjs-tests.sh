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

symlink_electron

export_node_bin_path

which electron
which node

node --version

deepspeech --version

check_runtime_electronjs

run_electronjs_inference_tests
