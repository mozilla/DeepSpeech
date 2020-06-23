#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

unset PYTHON_BIN_PATH
unset PYTHONPATH

export PATH="${PYENV_ROOT}/bin:$PATH"

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

for pyver_conf in ${SUPPORTED_PYTHON_VERSIONS}; do
  pyver=$(echo "${pyver_conf}" | cut -d':' -f1)
  pyconf=$(echo "${pyver_conf}" | cut -d':' -f2)

  pyalias="${pyver}_${pyconf}"

  PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf}" pyenv_install ${pyver} ${pyalias}

  setup_pyenv_virtualenv "${pyalias}" "deepspeech"

  virtualenv_activate "${pyalias}" "deepspeech"

  python --version
  python3 --version || true # Might fail without any issue on Windows
  which pip
  which pip3 || true # Might fail without any issue on Windows

  virtualenv_deactivate "${pyalias}" "deepspeech"
done;
