#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2
set_ldc_sample_filename "${bitrate}"

download_data

virtualenv_activate "${pyalias}" "deepspeech"

if [ "$3" = "cuda" ]; then
  deepspeech_pkg_url=$(get_python_pkg_url "${pyver_pkg}" "${py_unicode_type}" "deepspeech_gpu")
else
  deepspeech_pkg_url=$(get_python_pkg_url "${pyver_pkg}" "${py_unicode_type}")
fi;

LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: --upgrade ${deepspeech_pkg_url} | cat

which deepspeech
deepspeech --version

ensure_cuda_usage "$3"

run_all_inference_tests

virtualenv_deactivate "${pyalias}" "deepspeech"
