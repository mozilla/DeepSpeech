#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf"

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-test/.pyenv"
export PATH="${PYENV_ROOT}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true

model_source=${DEEPSPEECH_PROD_MODEL}
model_name=$(basename "${model_source}")

model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP}
model_name_mmap=$(basename "${model_source_mmap}")

download_data

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

maybe_ssl102_py37 ${pyver}

maybe_numpy_min_version_winamd64 ${pyver}

PYENV_NAME=deepspeech-test
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf} ${PY37_OPENSSL} ${EXTRA_PYTHON_CONFIGURE_OPTS}" pyenv_install ${pyver}

setup_pyenv_virtualenv "${pyver}" "${PYENV_NAME}"
virtualenv_activate "${pyver}" "${PYENV_NAME}"

deepspeech_pkg_url=$(get_python_pkg_url ${pyver_pkg} ${py_unicode_type})
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: ${PY37_SOURCE_PACKAGE} --upgrade ${deepspeech_pkg_url} | cat

run_prod_inference_tests

run_prod_concurrent_stream_tests

virtualenv_deactivate "${pyver}" "${PYENV_NAME}"
