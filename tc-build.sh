#!/bin/bash

set -xe

source ${HOME}/DeepSpeech/tf/tc-vars.sh

DS_TFDIR=${HOME}/DeepSpeech/tf
EXTRA_CUDA_CFLAGS=
EXTRA_CUDA_LDFLAGS=

if [ "$1" = "--gpu" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
    BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_OPT_FLAGS}"
    SYSTEM_TARGET=host
    EXTRA_CUDA_CFLAGS=-L${HOME}/DeepSpeech/CUDA/lib64/
    EXTRA_CUDA_LDFLAGS=-lcudart
fi

if [ "$1" = "--arm" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
    BAZEL_BUILD_FLAGS="${BAZEL_ARM_FLAGS}"
    SYSTEM_TARGET=rpi3
fi

if [ "$1" != "--gpu" -a "$1" != "--arm" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
    BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS}"
    SYSTEM_TARGET=host
    MAKE_BINDINGS_PY=1
    MAKE_BINDINGS_JS=1
fi

cd ~/DeepSpeech/tf
eval "export ${BAZEL_ENV_FLAGS}"
PATH=${HOME}/bin/:$PATH bazel \
	build -c opt ${BAZEL_BUILD_FLAGS} \
	//native_client:*

cd ~/DeepSpeech/ds/
make -C native_client/ \
	TARGET=${SYSTEM_TARGET} \
	TFDIR=${DS_TFDIR} \
	RASPBIAN=/tmp/multistrap-raspbian-jessie \
	EXTRA_CFLAGS=${EXTRA_CUDA_CFLAGS} \
	EXTRA_LDFLAGS=${EXTRA_CUDA_LDFLAGS} \
	deepspeech

if [ ${MAKE_BINDINGS_PY} ]; then
    unset PYTHON_BIN_PATH
    unset PYTHONPATH
    export PYENV_ROOT="${HOME_CLEAN}/DeepSpeech/.pyenv"
    export PATH="${PYENV_ROOT}/bin:$PATH"

    git clone --quiet https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
    pushd ${PYENV_ROOT}
        git checkout --quiet 0c909f7457a027276a1d733d78bfbe70ba652047
    popd
    eval "$(pyenv init -)"

    PYENV_VENV="$(pyenv root)/plugins/pyenv-virtualenv"
    git clone --quiet https://github.com/pyenv/pyenv-virtualenv.git ${PYENV_VENV}
    pushd ${PYENV_VENV}
        git checkout --quiet 27270877575fe8c3e7be5385b8b6a1e4089b39aa
    popd
    #eval "$(pyenv virtualenv-init -)"

    mkdir -p wheels

    for pyver in 2.7.13 3.4.6 3.5.3 3.6.2; do
        pyenv install ${pyver}
        pyenv virtualenv ${pyver} deepspeech
        source ${PYENV_ROOT}/versions/${pyver}/envs/deepspeech/bin/activate

        make -C native_client/ \
            TFDIR=${DS_TFDIR} \
            bindings

        cp native_client/dist/deepspeech-*.whl wheels

        make -C native_client/ bindings-clean

        deactivate
        pyenv uninstall --force deepspeech
    done;
fi

if [ ${MAKE_BINDINGS_JS} ]; then
    # 7.10.0 and 8.0.0 targets fails to build
    # > ../deepspeech_wrap.cxx:966:23: error: 'WeakCallbackData' in namespace 'v8' does not name a type
    for node in 4.8.0 5.12.0 6.10.0; do
        make -C native_client/javascript \
            TFDIR=${DS_TFDIR} \
            NODE_ABI_TARGET=--target=$node \
            clean package
    done;

    make -C native_client/javascript clean npm-pack
fi
