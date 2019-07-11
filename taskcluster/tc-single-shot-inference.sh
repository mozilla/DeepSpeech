#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

ds=$2
frozen=$2

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-train/.pyenv"
export PATH="${PYENV_ROOT}/bin:${HOME}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true
mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true
mkdir -p /tmp/train_tflite || true

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-train
PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf}" pyenv_install ${pyver}

setup_pyenv_virtualenv "${pyver}" "${PYENV_NAME}"
virtualenv_activate "${pyver}" "${PYENV_NAME}"

pip install --upgrade -r ${HOME}/DeepSpeech/ds/requirements.txt | cat

decoder_pkg_url=$(get_python_pkg_url ${pyver_pkg} ${py_unicode_type} "ds_ctcdecoder" "${DECODER_ARTIFACTS_ROOT}")
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: ${PY37_SOURCE_PACKAGE} --upgrade ${decoder_pkg_url} | cat

pushd ${HOME}/DeepSpeech/ds/
    time ./bin/run-tc-ldc93s1_singleshotinference.sh
popd

virtualenv_deactivate "${pyver}" "${PYENV_NAME}"
