#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

DS_TFDIR=${DS_ROOT_TASK}/DeepSpeech/tf
EXTRA_CUDA_CFLAGS=
EXTRA_CUDA_LDFLAGS=

if [ "$1" = "--gpu" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
    BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_OPT_FLAGS}"
    SYSTEM_TARGET=host
    EXTRA_CUDA_CFLAGS="-L${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/ -L${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/"
    EXTRA_CUDA_LDFLAGS="-lcudart -lcuda"
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

cd ${DS_ROOT_TASK}/DeepSpeech/tf
eval "export ${BAZEL_ENV_FLAGS}"
PATH=${DS_ROOT_TASK}/bin/:$PATH bazel ${BAZEL_OUTPUT_USER_ROOT} \
	build -c opt ${BAZEL_BUILD_FLAGS} \
	//native_client:deepspeech \
	//native_client:deepspeech_utils \
	//native_client:ctc_decoder_with_kenlm \
	//native_client:generate_trie

cd ${DS_ROOT_TASK}/DeepSpeech/ds/
make -C native_client/ \
	TARGET=${SYSTEM_TARGET} \
	TFDIR=${DS_TFDIR} \
	RASPBIAN=/tmp/multistrap-raspbian-jessie \
	EXTRA_CFLAGS="${EXTRA_CUDA_CFLAGS}" \
	EXTRA_LDFLAGS="${EXTRA_CUDA_LDFLAGS}" \
	deepspeech

if [ ${MAKE_BINDINGS_PY} ]; then
    unset PYTHON_BIN_PATH
    unset PYTHONPATH
    export PYENV_ROOT="${DS_ROOT_TASK}/DeepSpeech/.pyenv"
    export PATH="${PYENV_ROOT}/bin:$PATH"

    install_pyenv "${PYENV_ROOT}"
    install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

    mkdir -p wheels

    for pyver in 2.7.13 3.4.6 3.5.3 3.6.2; do
        pyenv install ${pyver}
        pyenv virtualenv ${pyver} deepspeech
        source ${PYENV_ROOT}/versions/${pyver}/envs/deepspeech/bin/activate

        make -C native_client/ \
            TFDIR=${DS_TFDIR} \
            bindings-clean bindings

        cp native_client/dist/deepspeech-*.whl wheels

        make -C native_client/ bindings-clean

        deactivate
        pyenv uninstall --force deepspeech
    done;
fi

if [ ${MAKE_BINDINGS_JS} ]; then
    npm update && npm install node-gyp node-pre-gyp

    export PATH="$(npm root)/.bin/:$PATH"

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
