#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2
set_ldc_sample_filename "${bitrate}"

download_data

virtualenv_activate "${pyalias}" "deepspeech"

deepspeech_pkg_url=$(get_python_pkg_url ${pyver_pkg} ${py_unicode_type})
set -o pipefail
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: --upgrade ${deepspeech_pkg_url} | cat
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pushd ${HOME}/DeepSpeech/ds
    pip install --upgrade . | cat
popd
set +o pipefail

which deepspeech
deepspeech --version

pushd ${HOME}/DeepSpeech/ds/
    python bin/import_ldc93s1.py data/smoke_test
    python evaluate_tflite.py --model "${TASKCLUSTER_TMP_DIR}/${model_name_mmap}" --scorer data/smoke_test/pruned_lm.scorer --csv data/smoke_test/ldc93s1.csv
popd

virtualenv_deactivate "${pyalias}" "deepspeech"
