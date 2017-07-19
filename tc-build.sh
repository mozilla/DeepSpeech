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
    make -C native_client/ \
        TFDIR=${DS_TFDIR} \
        bindings
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
