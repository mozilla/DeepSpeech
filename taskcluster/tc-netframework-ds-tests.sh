#!/bin/bash

set -xe

bitrate=$1
package_option=$2

source $(dirname "$0")/tc-tests-utils.sh

bitrate=$1
set_ldc_sample_filename "${bitrate}"

if [ "${package_option}" = "cuda" ]; then
    PROJECT_NAME="Mozilla-Voice-STT-CUDA"
elif [ "${package_option}" = "--tflite" ]; then
    PROJECT_NAME="Mozilla-Voice-STT-TFLite"
    model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
    model_name=$(basename "${model_source}")
    model_name_mmap=$(basename "${model_source}")
else
    PROJECT_NAME="Mozilla-Voice-STT"
fi

download_data

install_nuget "${PROJECT_NAME}"

DS_BINARY_FILE="MozillaVoiceSttConsole.exe"
ensure_cuda_usage "$2"

run_netframework_inference_tests
