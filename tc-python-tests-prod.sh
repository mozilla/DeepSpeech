#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

pyver=$1

if [ -z "${pyver}" ]; then
    echo "No python version given, aborting."
    exit 1
fi;

# 2.7.x => 27
pyver_pkg=$(echo "${pyver}" | cut -d'.' --output-delimiter="" -f1,2)

# mu => unicode, 2 bytes python 2.7
# m  => unicode, 4 bytes python >+ 3
py_unicode_type="m"
if [ "${pyver_pkg}" = "27" ]; then
    py_unicode_type="mu"
fi;

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-test/.pyenv"
export PATH="${PYENV_ROOT}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true

model_source=${DEEPSPEECH_PROD_MODEL}
model_name=$(basename "${model_source}")

download_data

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-test
pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

platform=$(python -c 'import sys; import platform; plat = platform.system().lower(); plat = "manylinux1" if plat == "linux" else plat; sys.stdout.write("%s_%s" % (plat, platform.machine()));')
deepspeech_pkg="deepspeech-0.0.2-cp${pyver_pkg}-cp${pyver_pkg}${py_unicode_type}-${platform}.whl"

pip install --upgrade scipy==0.19.1 ${DEEPSPEECH_ARTIFACTS_ROOT}/${deepspeech_pkg}

phrase_pbmodel_withlm=$(python ${HOME}/DeepSpeech/ds/native_client/client.py /tmp/${model_name} /tmp/LDC93S1.wav /tmp/alphabet.txt /tmp/lm.binary /tmp/trie)
assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}"

deactivate
pyenv uninstall --force ${PYENV_NAME}
