#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

pyver=$1

if [ -z "${pyver}" ]; then
    echo "No python version given, aborting."
    exit 1
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
pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

pip install --upgrade ${TENSORFLOW_WHEEL}
grep -v "tensorflow" ${HOME}/DeepSpeech/ds/requirements.txt | pip install --upgrade -r /dev/stdin

download_native_client_files "/tmp/ds"

pushd ${HOME}/DeepSpeech/ds/
    time ./bin/run-tc-ldc93s1.sh
popd

deactivate
pyenv uninstall --force ${PYENV_NAME}

cp /tmp/train/output_graph.pb ${TASKCLUSTER_ARTIFACTS}
