#!/bin/bash

set -xe

#TODO: Remove after 0.6.1
export DEEPSPEECH_PROD_MODEL=https://github.com/lissyx/DeepSpeech/releases/download/v0.6.0/output_graph.tflite
export DEEPSPEECH_PROD_MODEL_MMAP=https://github.com/lissyx/DeepSpeech/releases/download/v0.6.0/output_graph.tflite

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2
set_ldc_sample_filename "${bitrate}"

unset PYTHON_BIN_PATH
unset PYTHONPATH

if [ -d "${DS_ROOT_TASK}/pyenv.cache/" ]; then
  export PYENV_ROOT="${DS_ROOT_TASK}/pyenv.cache/ds-test/.pyenv"
else
  export PYENV_ROOT="${DS_ROOT_TASK}/ds-test/.pyenv"
fi;

export PATH="${PYENV_ROOT}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true

model_source=${DEEPSPEECH_PROD_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")
model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP//.pbmm/.tflite}
export DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_TFLITE_ROOT}

download_data

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

maybe_ssl102_py37 ${pyver}

maybe_numpy_min_version_winamd64 ${pyver}

PYENV_NAME=deepspeech-test
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf} ${PY37_OPENSSL} ${EXTRA_PYTHON_CONFIGURE_OPTS}" pyenv_install ${pyver} ${pyalias}

setup_pyenv_virtualenv "${pyalias}" "${PYENV_NAME}"
virtualenv_activate "${pyalias}" "${PYENV_NAME}"

deepspeech_pkg_url=$(get_python_pkg_url "${pyver_pkg}" "${py_unicode_type}" "deepspeech_tflite")
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: ${PY37_SOURCE_PACKAGE} --upgrade ${deepspeech_pkg_url} | cat

which deepspeech
deepspeech --version

run_prodtflite_inference_tests "${bitrate}"

virtualenv_deactivate "${pyalias}" "${PYENV_NAME}"
