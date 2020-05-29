#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true
mkdir -p /tmp/train_tflite || true

virtualenv_activate "${pyalias}" "deepspeech"

set -o pipefail
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pushd ${HOME}/DeepSpeech/ds
    pip install --upgrade . | cat
popd
set +o pipefail

python -m unittest
