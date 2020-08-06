#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")
model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP//.pbmm/.tflite}

if [ ! -z "${DEEPSPEECH_ARTIFACTS_TFLITE_ROOT}" ]; then
    export DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_TFLITE_ROOT}
fi;

download_data

maybe_setup_virtualenv_cross_arm "${pyalias}" "deepspeech"

virtualenv_activate "${pyalias}" "deepspeech"

pkg_name=$(get_tflite_python_pkg_name)
deepspeech_pkg_url=$(get_python_pkg_url "${pyver_pkg}" "${py_unicode_type}" "${pkg_name}")
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: --upgrade ${deepspeech_pkg_url} | cat

which mozilla_voice_stt
mozilla_voice_stt --version

run_all_inference_tests

virtualenv_deactivate "${pyalias}" "deepspeech"
