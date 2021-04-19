#!/bin/bash

set -xe

source $(dirname "$0")/all-vars.sh
source $(dirname "$0")/all-utils.sh
source $(dirname "$0")/asserts.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_PROD_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")
model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP//.pbmm/.tflite}

download_model_prod

download_material

export_py_bin_path

deepspeech --version

run_prodtflite_inference_tests "${bitrate}"
