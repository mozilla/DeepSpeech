#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

pyver_full=$1
ds=$2
frozen=$2

if [ -z "${pyver_full}" ]; then
    echo "No python version given, aborting."
    exit 1
fi;

pyver=$(echo "${pyver_full}" | cut -d':' -f1)

# 2.7.x => 27
pyver_pkg=$(echo "${pyver}" | cut -d'.' -f1,2 | tr -d '.')

py_unicode_type=$(echo "${pyver_full}" | cut -d':' -f2)
if [ "${py_unicode_type}" = "m" ]; then
  pyconf="ucs2"
elif [ "${py_unicode_type}" = "mu" ]; then
  pyconf="ucs4"
fi;

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-train/.pyenv"
export PATH="${PYENV_ROOT}/bin:${HOME}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true
mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-train
PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf}" time pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

pip install --upgrade -r ${HOME}/DeepSpeech/ds/requirements.txt | cat

download_ctc_kenlm "/tmp/ds"

pushd ${HOME}/DeepSpeech/ds/
    time ./bin/run-tc-ldc93s1_singleshotinference.sh
popd

deactivate
pyenv uninstall --force ${PYENV_NAME}
