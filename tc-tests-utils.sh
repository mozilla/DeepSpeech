#!/bin/bash

set -xe

OS=$(uname)
if [ "${OS}" = "Linux" ]; then
    export DS_ROOT_TASK=${HOME}
fi;

if [ "${OS}" = "Darwin" ]; then
    export DS_ROOT_TASK=${TASKCLUSTER_TASK_DIR}
fi;

export TASKCLUSTER_ARTIFACTS=${TASKCLUSTER_ARTIFACTS:-/tmp/artifacts}

model_name=$(basename "${DEEPSPEECH_MODEL}")

assert_correct_inference()
{
  phrase=$1
  expected=$2

  if [ -z "${phrase}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "phrase: <${phrase}>"
      echo "expected: <${expected}>"
      exit 1;
  fi;

  if [ "${phrase}" = "${expected}" ]; then
      echo "Proper output has been produced:"
      echo "${phrase}"
      exit 0
  else
      echo "!! Non matching output !!"
      echo "got: <${phrase}>"
      echo "xxd:"; echo "${phrase}" | xxd
      echo "-------------------"
      echo "expected: <${expected}>"
      echo "xxd:"; echo "${expected}" | xxd
      exit 1
  fi;
}

assert_correct_ldc93s1()
{
  assert_correct_inference "$1" "she had your dark suit in greasy wash water all year"
}

download_material()
{
  target_dir=$1

  if [ -z "${target_dir}" ]; then
    echo "Empty name for target directory: ${target_dir}"
    exit 1
  fi;

  mkdir -p ${target_dir} || true

  wget ${DEEPSPEECH_MODEL} -O /tmp/${model_name}
  wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav -O /tmp/LDC93S1.wav
  wget ${DEEPSPEECH_ARTIFACTS_ROOT}/native_client.tar.xz -O - | pixz -d | tar -C ${target_dir} -xf -

  ls -hal /tmp/${model_name} /tmp/LDC93S1.wav
}

install_pyenv()
{
  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  git clone --quiet https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
  pushd ${PYENV_ROOT}
    git checkout --quiet 0c909f7457a027276a1d733d78bfbe70ba652047
  popd
  eval "$(pyenv init -)"
}

install_pyenv_virtualenv()
{
  PYENV_VENV=$1

  if [ -z "${PYENV_VENV}" ]; then
    echo "No PYENV_VENV set";
    exit 1;
  fi;

  git clone --quiet https://github.com/pyenv/pyenv-virtualenv.git ${PYENV_VENV}
  pushd ${PYENV_VENV}
      git checkout --quiet 27270877575fe8c3e7be5385b8b6a1e4089b39aa
  popd
  eval "$(pyenv virtualenv-init -)"
}
