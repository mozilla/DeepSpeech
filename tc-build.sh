#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

DS_TFDIR=${DS_ROOT_TASK}/DeepSpeech/tf
EXTRA_CUDA_CFLAGS=
EXTRA_CUDA_LDFLAGS=

BAZEL_AOT_BUILD_FLAGS="--define=DS_MODEL_TIMESTEPS=64 --define=DS_MODEL_FRAMESIZE=494 --define=DS_MODEL_FILE=${DS_ROOT_TASK}/DeepSpeech/ds/test.frozen.494_e50_master.LSTM.ldc93s1.pb"
BAZEL_AOT_EXTRA_TARGETS="//native_client:deepspeech_model //tensorflow/compiler/aot:runtime //tensorflow/compiler/xla/service/cpu:runtime_matmul //tensorflow/compiler/xla:executable_run_options"
EXTRA_AOT_CFLAGS=""
EXTRA_AOT_LIBS=" -ldeepspeech_model -lruntime -lruntime_matmul -lexecutable_run_options"
EXTRA_AOT_LDFLAGS="-L${DS_TFDIR}/bazel-bin/tensorflow/compiler/xla -L${DS_TFDIR}/bazel-bin/tensorflow/compiler/aot -L${DS_TFDIR}/bazel-bin/tensorflow/compiler/xla/service/cpu"

if [ "$1" = "--gpu" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
    BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_OPT_FLAGS}"
    SYSTEM_TARGET=host
    EXTRA_CUDA_CFLAGS=""
    EXTRA_CUDA_LDFLAGS="-L${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/ -L${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/ -lcudart -lcuda"

    # Actually reset those, we don't care that much about tfcompile with CUDA support:
    # I tensorflow/compiler/xla/service/platform_util.cc:72] platform CUDA present but no XLA compiler available: could not find registered compiler for platform CUDA -- check target linkage
    BAZEL_AOT_EXTRA_TARGETS=""
    EXTRA_AOT_CFLAGS=""
    EXTRA_AOT_LDFLAGS=""
    EXTRA_AOT_LIBS=""
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
	build -c opt ${BAZEL_BUILD_FLAGS} ${BAZEL_AOT_BUILD_FLAGS} \
	//native_client:deepspeech \
	//native_client:deepspeech_utils \
	//native_client:ctc_decoder_with_kenlm \
	//native_client:generate_trie \
	${BAZEL_AOT_EXTRA_TARGETS}

cd ${DS_ROOT_TASK}/DeepSpeech/ds/
make -C native_client/ \
	TARGET=${SYSTEM_TARGET} \
	TFDIR=${DS_TFDIR} \
	RASPBIAN=/tmp/multistrap-raspbian-jessie \
	EXTRA_CFLAGS="${EXTRA_CUDA_CFLAGS} ${EXTRA_AOT_CFLAGS}" \
	EXTRA_LDFLAGS="${EXTRA_CUDA_LDFLAGS} ${EXTRA_AOT_LDFLAGS}" \
	EXTRA_LIBS="${EXTRA_AOT_LIBS}" \
	deepspeech

if [ "${OS}" = "Darwin" ]; then
    export SWIG_LIB="$(find ${DS_ROOT_TASK}/homebrew/Cellar/swig/ -type f -name "swig.swg" | xargs dirname)"
fi;

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

        EXTRA_CFLAGS="${EXTRA_AOT_CFLAGS}" EXTRA_LDFLAGS="${EXTRA_AOT_LDFLAGS}" EXTRA_LIBS="${EXTRA_AOT_LIBS}" make -C native_client/ \
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
        EXTRA_CFLAGS="${EXTRA_AOT_CFLAGS}" EXTRA_LDFLAGS="${EXTRA_AOT_LDFLAGS}" EXTRA_LIBS="${EXTRA_AOT_LIBS}" make -C native_client/javascript \
            TFDIR=${DS_TFDIR} \
            NODE_ABI_TARGET=--target=$node \
            clean package
    done;

    make -C native_client/javascript clean npm-pack
fi
