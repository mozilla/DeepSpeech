#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

pyver=$1
aot_model=$2

if [ -z "${pyver}" ]; then
    echo "No python version given, aborting."
    exit 1
fi;

# 2.7.x => 27
pyver_pkg=$(echo "${pyver}" | cut -d'.' -f1,2 | tr -d '.')

# mu => unicode, 2 bytes python 2.7
# m  => unicode, 4 bytes python >+ 3
py_unicode_type="m"
if [ "${OS}" = "Linux" -a "${pyver_pkg}" = "27" ]; then
    py_unicode_type="mu"
fi;

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${DS_ROOT_TASK}/ds-test/.pyenv"
export PATH="${PYENV_ROOT}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true

download_data

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-test
pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

platform=$(python -c 'import sys; import platform; plat = platform.system().lower(); plat = "manylinux1" if plat == "linux" else plat; plat = "macosx_10_10" if plat == "darwin" else plat; sys.stdout.write("%s_%s" % (plat, platform.machine()));')
deepspeech_pkg="deepspeech-0.1.1-cp${pyver_pkg}-cp${pyver_pkg}${py_unicode_type}-${platform}.whl"

if [ "${aot_model}" = "--aot" ]; then
    deepspeech_pkg_url=${DEEPSPEECH_AOT_ARTIFACTS_ROOT}/${deepspeech_pkg}
else
    deepspeech_pkg_url=${DEEPSPEECH_ARTIFACTS_ROOT}/${deepspeech_pkg}
fi
pip install --upgrade ${deepspeech_pkg_url} | cat

run_all_inference_tests

deactivate
pyenv uninstall --force ${PYENV_NAME}
