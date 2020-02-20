#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true
mkdir -p /tmp/train_tflite || true

virtualenv_activate "${pyalias}" "deepspeech"

set -o pipefail
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pip install --upgrade -r ${HOME}/DeepSpeech/ds/requirements.txt | cat
set +o pipefail

pushd ${HOME}/DeepSpeech/ds/
    verify_ctcdecoder_url
popd

decoder_pkg_url=$(get_python_pkg_url ${pyver_pkg} ${py_unicode_type} "ds_ctcdecoder" "${DECODER_ARTIFACTS_ROOT}")
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: ${PY37_SOURCE_PACKAGE} ${decoder_pkg_url} | cat

pushd ${HOME}/DeepSpeech/ds/
    time ./bin/run-tc-transfer.sh
popd

virtualenv_deactivate "${pyalias}" "deepspeech"
